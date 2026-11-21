"""Frozen VideoMAEv2 H -> trainable S + one Linear, last-block token MSE."""
import argparse
import copy
import gc
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .common import ROOT, ASSETS, SETTINGS, sha, save as atomic_json, check_setup
from .vit_encoder import build_encoder, forward_tokens, tensor_hash
from .vit_data import Videos, inputs, loader, seed_all, schedule_factor
from .state import save, student_tokens, rng_state, restore_rng

LOG = logging.getLogger('token_distillation')






def alignment(student, projector, teacher, x, use_checkpoint):
    # no_grad (rather than inference_mode) leaves ordinary target tensors that
    # autograd can save for the MSE backward. Exactly the SAME x goes to H and S.
    teacher.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        target = forward_tokens(teacher, x)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        source = student_tokens(student, x, use_checkpoint)
        prediction = projector(source)
    assert tuple(source.shape[1:]) == (1568, 384)
    assert prediction.shape == target.shape and tuple(target.shape[1:]) == (1568, 1280)
    error = (prediction.float() - target.float()).square().mean((1, 2))
    with torch.no_grad():
        cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1).mean(1)
        energy = target.float().square().mean((1, 2))
    return error, cosine, energy


def groups(student, projector, args):
    result = {}
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        layer = int(name.split('.')[1]) + 1 if name.startswith('blocks.') else 0
        decay = args.weight_decay if p.ndim > 1 else 0.
        scale = args.layer_decay ** (12 - layer)
        key = (layer, decay)
        result.setdefault(key, dict(params=[], base_lr=args.lr * scale, lr=args.lr * scale,
                                    weight_decay=decay, name=f'student_layer{layer}_wd{decay}'))['params'].append(p)
    for name, p in projector.named_parameters():
        result[('linear', name)] = dict(params=[p], base_lr=args.linear_lr, lr=args.linear_lr,
                                      weight_decay=args.weight_decay if p.ndim > 1 else 0., name=f'linear_{name}')
    return list(result.values())


def empty_totals():
    return dict(mse=0., cosine=0., teacher_energy=0., clips=0)


def add(totals, error, cosine, energy):
    totals['mse'] += float(error.detach().sum())
    totals['cosine'] += float(cosine.sum())
    totals['teacher_energy'] += float(energy.sum())
    totals['clips'] += len(error)


def average(totals):
    count = max(1, totals['clips'])
    values = {k: totals[k] / count for k in ['mse', 'cosine', 'teacher_energy']}
    values['relative_mse'] = totals['mse'] / max(totals['teacher_energy'], 1e-12)
    values['modality_clips'] = totals['clips']
    return values


@torch.no_grad()
def evaluate(student, projector, teacher, batches):
    student.eval()
    projector.eval()
    totals = [empty_totals(), empty_totals()]
    rows = []
    begun = last_log = time.perf_counter()
    for _, (indices, videos, _) in enumerate(batches):
        for i in range(len(indices)):
            row = dict(index=int(indices[i]))
            for stream, name in enumerate(['depth', 'ir']):
                x = videos[i:i + 1, stream].cuda(non_blocking=True)
                error, cosine, energy = alignment(student, projector, teacher, x, False)
                add(totals[stream], error, cosine, energy)
                row[name] = dict(mse=float(error[0]), cosine=float(cosine[0]), teacher_energy=float(energy[0]))
            rows.append(row)
        now = time.perf_counter()
        if now - last_log >= 25 or len(rows) == len(batches.dataset):
            LOG.info('validation %d/%d %.1fs', len(rows), len(batches.dataset), now - begun)
            last_log = now
    combined = {k: totals[0][k] + totals[1][k] for k in totals[0]}
    return dict(overall=average(combined), depth=average(totals[0]), ir=average(totals[1])), rows


def draw(out, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    train = [r for r in history if 'train' in r]
    val = [r for r in history if 'validation' in r]
    for ax, key, label in zip(axes, ['mse', 'cosine'], ['Token MSE (lower is better)', 'Token cosine (higher is better)']):
        ax.plot([r['epoch'] for r in train], [r['train'][key] for r in train], label='train', alpha=.8)
        for mode in ['overall', 'ir', 'depth']:
            ax.plot([r['epoch'] for r in val], [r['validation'][mode][key] for r in val], marker='o', markersize=3,
                    label=f'validation {mode}')
        ax.set(xlabel='Epoch', ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out / 'distillation_curves.png', dpi=160)
    fig.savefig(out / 'distillation_curves.svg')
    plt.close(fig)


def write_history(out, history):
    path = out / 'training.jsonl'
    temp = path.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r) + '\n' for r in history), encoding='utf8')
    temp.replace(path)
    draw(out, history)






def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/distill')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--check', action='store_true', help='Check H/S weights and dataset only; no training')
    parser.add_argument('--epochs', type=int, default=SETTINGS['distill']['epochs'])
    parser.add_argument('--val-every', type=int, default=SETTINGS['distill']['val_every'])
    parser.add_argument('--batch-size', type=int, default=1, help='paired IR+Depth videos per loader batch')
    parser.add_argument('--effective-batch-size', type=int, default=SETTINGS['distill']['effective_batch_size'], help='individual modality clips per optimizer update')
    parser.add_argument('--lr', type=float, default=SETTINGS['distill']['lr'])
    parser.add_argument('--linear-lr', type=float, default=SETTINGS['distill']['linear_lr'])
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--layer-decay', type=float, default=.75)
    parser.add_argument('--warmup-epochs', type=int, default=SETTINGS['distill']['warmup_epochs'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every-updates', type=int, default=100)
    parser.add_argument('--no-gradient-checkpointing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_setup('distill',args.data_root)));return
    if min(args.epochs, args.val_every, args.batch_size, args.effective_batch_size, args.lr, args.linear_lr,
           args.save_every_updates) <= 0 or args.workers < 0 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error('Invalid numeric parameters')
    if args.effective_batch_size % (2 * args.batch_size) or not 0 < args.layer_decay <= 1 or args.weight_decay < 0:
        parser.error('Effective batch must be divisible by 2*batch; invalid layer/weight decay')
    out = args.output_dir.resolve()
    if out == ROOT / 'results' or not out.is_relative_to(ROOT / 'results'):
        parser.error('Use a separate directory under vit/results')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Old outputs protected: choose a new --output-dir or --resume')
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(sys.stdout),
                        logging.FileHandler(out / 'run.log', mode='a' if args.resume else 'w', encoding='utf8')])
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(args.seed)
    root, records, provenance, classes, protected = inputs(args.data_root,args.seed)
    for path in [*Path(__file__).parent.glob('*.py'), ROOT/'upstream/videomaev2/modeling_finetune.py']:
        protected[str(path.resolve())] = sha(path)
    for key in ['vit','teacher']:
        asset=ASSETS[key]
        protected[str(ROOT/asset['file'])]=asset['sha256']
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ['resume', 'output_dir']}
    config.update(data_root=str(root), train_samples=1995, validation_samples=466, frames=16, size=224,
                  objective='raw last-block token MSE, no target normalization, no labels or classification head',
                  architecture='shared S on IR and Depth separately; Linear(384,1280,bias=True); frozen H eval/no_grad',
                  teacher_provenance='official mae-h checkpoint is UCF101 fine-tuned; metadata32 frames adapted to16',
                  student_provenance='official giant-distilled K710 S; pretrained initialization, not random',
                  token_shape=[1568, 384], aligned_shape=[1568, 1280], projection_parameters=492800,
                  unused='S pretrained fc_norm retained frozen for compatibility; pooling/norm/classification never called',
                  precision='FP32 parameters + BF16 autocast + FP32 MSE; sequential modality microbatch1',
                  augmentation='same tensor to H/S, epoch-random16 frames/crop/flip paired between modalities',
                  optimizer='AdamW betas(.9,.999) eps1e-8, no decay for bias/norm; cosine to1%, clip_grad1',
                  best='lowest overall fixed validation token MSE, eval epoch0 and every5 (configurable)',
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0), source_fingerprints=protected)
    if args.resume:
        assert json.loads((out / 'config.json').read_text(encoding='utf8')) == config, 'Configuration/source changed'
    else:
        atomic_json(out / 'config.json', config)
    LOG.info('Loading H and S official backbones; no MLP/classification head; GPU=%s', config['gpu'])
    teacher, teacher_info = build_encoder('huge')
    teacher_before = tensor_hash(teacher.state_dict())
    teacher.cuda().requires_grad_(False).eval()
    student, student_info = build_encoder('small')
    student.cuda().requires_grad_(True)
    student.fc_norm.requires_grad_(False)  # not on the pre-pooling token path
    projector = nn.Linear(384, 1280, bias=True).cuda()
    nn.init.xavier_uniform_(projector.weight)
    nn.init.zeros_(projector.bias)
    trainable = [p for m in [student, projector] for p in m.parameters() if p.requires_grad]
    assert sum(p.numel() for p in projector.parameters()) == 492800
    assert sum(p.numel() for p in trainable) == 22371968
    optimizer = torch.optim.AdamW(groups(student, projector, args), betas=(.9, .999), eps=1e-8)
    train = Videos(records['train'], root, 'joint', True, args.seed)
    validation = Videos(records['validation'], root, 'joint', False, args.seed)
    val_batches = loader(validation, args.batch_size, args.workers)
    first_epoch, next_batch, step = 1, 0, 0
    best, history, totals = None, [], empty_totals()
    resume_rng = None
    last_path = out / 'checkpoints/last.pt'
    if args.resume:
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        assert saved['config'] == config and saved['teacher_tensor_sha256'] == teacher_before
        student.load_state_dict(saved['student'], strict=True)
        projector.load_state_dict(saved['linear'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        first_epoch, next_batch, step = saved['next_epoch'], saved['next_batch'], saved['step']
        history, best, totals = saved['history'], saved['best'], saved['partial_train_totals']
        resume_rng = saved['rng']
        del saved
        LOG.info('Resume epoch=%d next_batch=%d optimizer_updates=%d', first_epoch, next_batch, step)
    atomic_json(out / 'model_info.json', dict(teacher=teacher_info, student=student_info,
                teacher_sha256=teacher_before, trainable_parameters=sum(p.numel() for p in trainable),
                student_token_parameters=21879168, student_unused_frozen_fc_norm_parameters=768,
                projection_parameters=492800))

    def state(epoch, next_epoch, batch_index, full=True):
        value = dict(epoch=epoch, next_epoch=next_epoch, next_batch=batch_index, step=step, config=config,
                     student={k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
                     linear={k: v.detach().cpu().clone() for k, v in projector.state_dict().items()},
                     teacher_tensor_sha256=teacher_before, best=copy.deepcopy(best), history=copy.deepcopy(history))
        if full:
            value.update(optimizer=optimizer.state_dict(), rng=rng_state(), partial_train_totals=copy.deepcopy(totals))
        return value

    if not history:
        metrics, _ = evaluate(student, projector, teacher, val_batches)
        best = dict(epoch=0, validation=metrics)
        history.append(dict(epoch=0, validation=metrics))
        save(state(0, 1, 0, False), out / 'checkpoints/best.pt')
        save(state(0, 1, 0), last_path)
        write_history(out, history)
        LOG.info('Initial validation MSE=%.6f cosine=%.4f', metrics['overall']['mse'], metrics['overall']['cosine'])
    accumulation = args.effective_batch_size // (2 * args.batch_size)
    batch_count = math.ceil(len(train) / args.batch_size)
    updates_per_epoch = math.ceil(batch_count / accumulation)
    for epoch in range(first_epoch, args.epochs + 1):
        train.epoch.value = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        batches = loader(train, args.batch_size, args.workers, generator=generator, shuffle=True)
        student.train()
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        if epoch != first_epoch or next_batch == 0:
            totals = empty_totals()
        # Recreate the epoch's deterministic order, skipping completed batches.
        iterator = iter(batches)
        for _ in range(next_batch):
            next(iterator)
        if resume_rng is not None:
            restore_rng(resume_rng)
            resume_rng = None
        begun = last_log = time.perf_counter()
        start_clips = totals['clips']
        factor = schedule_factor(max(0, step - 1), updates_per_epoch * args.epochs,
                                 updates_per_epoch * args.warmup_epochs)
        for batch_index in range(next_batch, batch_count):
            _, videos, _ = next(iterator)  # class labels intentionally unused
            group_start = batch_index // accumulation * accumulation
            group_pairs = min(accumulation * args.batch_size, len(train) - group_start * args.batch_size)
            factor = schedule_factor(step, updates_per_epoch * args.epochs, updates_per_epoch * args.warmup_epochs)
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            flat = videos.flatten(0, 1)
            for clip in flat.split(1):
                error, cosine, energy = alignment(student, projector, teacher, clip.cuda(non_blocking=True),
                                                 not args.no_gradient_checkpointing)
                loss = error.sum() / (2 * group_pairs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss epoch{epoch} batch{batch_index}')
                loss.backward()
                add(totals, error, cosine, energy)
            boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count
            if boundary:
                norm = nn.utils.clip_grad_norm_(trainable, 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient norm')
                if step == 0:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
                    assert not teacher.training and all(p.grad is None and not p.requires_grad for p in teacher.parameters())
                    atomic_json(out / 'gradient_verification.json', dict(passed=True, teacher_frozen=True,
                                all_student_token_and_projection_parameters_receive_gradients=True,
                                student_unused_fc_norm_frozen=True, gradient_norm=float(norm)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.save_every_updates == 0:
                    save(state(epoch, epoch, batch_index + 1), last_path)
            now = time.perf_counter()
            if now - last_log >= 25 or batch_index + 1 == batch_count:
                rate = (totals['clips'] - start_clips) / max(now - begun, 1e-6)
                mean = average(totals)
                LOG.info('epoch%d/%d batch%d/%d MSE=%.6f cosine=%.4f %.2fclips/s epochETA=%.0fs peak=%.2fGB',
                         epoch, args.epochs, batch_index + 1, batch_count, mean['mse'], mean['cosine'], rate,
                         (len(train) * 2 - totals['clips']) / max(rate, 1e-6), torch.cuda.max_memory_allocated() / 1e9)
                last_log = now
        assert totals['clips'] == len(train) * 2
        row = dict(epoch=epoch, train=average(totals), lr=args.lr * factor, linear_lr=args.linear_lr * factor,
                   optimizer_updates=step, training_seconds=time.perf_counter() - begun)
        # Save pending-evaluation state FIRST: an interrupted eval can be rerun
        # without repeating the already completed training epoch.
        save(state(epoch, epoch, batch_count), last_path)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics, _ = evaluate(student, projector, teacher, val_batches)
            row['validation'] = metrics
            row['saved_as_best'] = metrics['overall']['mse'] < best['validation']['overall']['mse']
            if row['saved_as_best']:
                best = dict(epoch=epoch, validation=metrics)
        history.append(row)
        if row.get('saved_as_best'):
            save(state(epoch, epoch + 1, 0, False), out / 'checkpoints/best.pt')
        totals = empty_totals()
        save(state(epoch, epoch + 1, 0), last_path)
        write_history(out, history)
        LOG.info('EPOCH COMPLETE %s', json.dumps(row))
        atomic_json(out / 'progress.json', dict(status='training', completed_epoch=epoch, epochs=args.epochs,
                    best=best, last=row, updated_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        next_batch = 0
        del batches, iterator
        gc.collect()
    saved = torch.load(out / 'checkpoints/best.pt', map_location='cpu', weights_only=True)
    student.load_state_dict(saved['student'], strict=True)
    projector.load_state_dict(saved['linear'], strict=True)
    del saved
    metrics, rows = evaluate(student, projector, teacher, val_batches)
    assert abs(metrics['overall']['mse'] - best['validation']['overall']['mse']) < 1e-5
    for row in rows:
        row.update(sample_id=records['validation'][row['index']]['sample_id'], epoch=best['epoch'])
    (out / 'validation_alignment.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
    teacher_after = tensor_hash(teacher.state_dict())
    assert teacher_after == teacher_before and all(p.grad is None for p in teacher.parameters())
    for name, expected in protected.items():
        assert sha(Path(name)) == expected, f'Protected input changed: {name}'
    result = dict(passed=True, best=best, teacher_unchanged=True, teacher_tensor_sha256=teacher_after,
                  source_files_unchanged=True, validation_pairs=len(rows), metrics=metrics,
                  completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(out / 'complete.json', result)
    atomic_json(out / 'progress.json', dict(status='complete', **result))
    LOG.info('COMPLETE best epoch%d validation MSE=%.6f cosine=%.4f; frozen H verified unchanged',
             best['epoch'], metrics['overall']['mse'], metrics['overall']['cosine'])


if __name__ == '__main__':
    mp.freeze_support()
    main()
