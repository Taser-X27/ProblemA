# 路径适配副本：训练算法保持原实现，允许结果写入当前项目results。
"""本次H→S蒸馏完成后，加载最佳学生，丢弃Linear，训练S＋随机MLP。"""
import argparse
import copy
import gc
import json
import logging
import math
import multiprocessing as mp
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, ASSETS, SETTINGS, sha, save as atomic_json, check_setup
from .vit_encoder import build_encoder, tensor_hash
from .vit_data import inputs, Videos, loader, metrics_with_subjects, seed_all, schedule_factor
from .state import student_tokens, save, rng_state, restore_rng

LOG = logging.getLogger('distilled_s_classifier')


class Classifier(nn.Module):
    def __init__(self, mode, args, load_initialization=True):
        super().__init__()
        self.mode = mode
        self.streams = 2 if mode == 'joint' else 1
        self.gradient_checkpointing = not args.no_gradient_checkpointing
        self.encoder_batch_size = args.encoder_batch_size
        self.backbone, _ = build_encoder('small')
        self.source_epoch = None
        if load_initialization:
            source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            assert 'student' in source and 'linear' in source,'需要H→S蒸馏checkpoint；先运行distill_vit.py'
            assert source['config']['aligned_shape']==[1568,1280]
            assert source['epoch']==source['best']['epoch']
            self.backbone.load_state_dict(source['student'],strict=True)
            self.source_epoch=source['epoch']
        # The discarded 384->1280 Linear is deliberately never constructed.
        # DropPath adds no learned parameters and does not change state keys.
        from timm.layers import DropPath
        for block, rate in zip(self.backbone.blocks, torch.linspace(0, args.drop_path, 12).tolist()):
            block.drop_path = DropPath(rate) if rate else nn.Identity()
        width = 384 * self.streams
        self.head = nn.Sequential(nn.LayerNorm(width, eps=1e-6), nn.Linear(width, args.hidden),
                                  nn.GELU(), nn.Dropout(args.dropout), nn.Linear(args.hidden, 40))
        for layer in self.head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.requires_grad_(True)

    def forward(self, videos):
        b, streams, c, t, h, w = videos.shape
        assert (streams, c, t, h, w) == (self.streams, 3, 16, 224, 224)
        pooled = []
        # A small S microbatch controls memory independently of paired video
        # batch size; modalities still share ONE backbone and its gradients.
        for chunk in videos.flatten(0, 1).split(self.encoder_batch_size):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens = student_tokens(self.backbone, chunk, self.gradient_checkpointing)
            with torch.autocast('cuda', enabled=False):
                pooled.append(self.backbone.fc_norm(tokens.float().mean(1)))
        with torch.autocast('cuda', enabled=False):
            return self.head(torch.cat(pooled).reshape(b, streams * 384))


def optimizer_groups(model, args):
    groups = {}
    for name, p in model.named_parameters():
        assert p.requires_grad
        if name.startswith('backbone.blocks.'):
            index = int(name.split('.')[2])
            lr = args.lr * args.layer_decay ** (11 - index)
        elif name.startswith('backbone.fc_norm.'):
            lr = args.lr
        elif name.startswith('backbone.'):
            lr = args.lr * args.layer_decay ** 12
        else:
            lr = args.head_lr
        decay = args.weight_decay if p.ndim > 1 else 0.
        groups.setdefault((lr, decay), dict(params=[], lr=lr, base_lr=lr, weight_decay=decay))['params'].append(p)
    grouped = [id(p) for g in groups.values() for p in g['params']]
    assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))
    return list(groups.values())


@torch.inference_mode()
def evaluate(model, batches, records, provenance, classes, smoothing):
    model.eval()
    logits, labels = torch.zeros(len(records), 40), torch.full((len(records),), -1, dtype=torch.long)
    seen = set()
    begun = last = time.perf_counter()
    for indices, videos, truth in batches:
        assert not seen.intersection(indices.tolist())
        logits[indices] = model(videos.cuda(non_blocking=True)).cpu()
        labels[indices] = truth
        seen.update(indices.tolist())
        now = time.perf_counter()
        if now - last >= 25 or len(seen) == len(records):
            LOG.info('%s validation %d/%d elapsed=%.1fs', model.mode, len(seen), len(records), now - begun)
            last = now
    assert seen == set(range(len(records))) and torch.isfinite(logits).all()
    assert labels.tolist() == [r['label'] for r in records]
    return metrics_with_subjects(labels, logits, classes, records, provenance, smoothing), logits.softmax(1)


def history_outputs(out, mode, history):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in history)
    temp = out / f'training_{mode}.jsonl.tmp'
    temp.write_text(text, encoding='utf8')
    temp.replace(out / f'training_{mode}.jsonl')
    train, val = [r for r in history if 'train_loss' in r], [r for r in history if 'validation' in r]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot([r['epoch'] for r in train], [r['train_loss'] for r in train], label='Train (augmented)')
    axes[0].plot([r['epoch'] for r in val], [r['validation']['loss_smoothed'] for r in val], 'o-', label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy (label smoothing)', ylim=(0, None))
    axes[1].plot([r['epoch'] for r in train], [100 * r['train_accuracy'] for r in train], label='Train (augmented)')
    axes[1].plot([r['epoch'] for r in val], [100 * r['validation']['accuracy'] for r in val], 'o-', label='Validation')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'official giant-distilled ViT-S + MLP | full fine-tuning | {mode}')
    for ext in ['png', 'svg']:
        fig.savefig(out / f'curves_{mode}.{ext}', dpi=160)
    plt.close(fig)
    with (out / f'curves_{mode}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'])
        writer.writeheader()
        for row in history:
            writer.writerow(dict(epoch=row['epoch'], train_loss=row.get('train_loss'), train_accuracy=row.get('train_accuracy'),
                                 validation_loss=row.get('validation', {}).get('loss_smoothed'),
                                 validation_accuracy=row.get('validation', {}).get('accuracy')))


def train_mode(mode, args, config, records, root, provenance, classes):
    out = args.output_dir
    if args.resume and (out / f'metrics_{mode}.json').exists():
        return json.loads((out / f'metrics_{mode}.json').read_text(encoding='utf8'))
    seed_all(args.seed)
    model = Classifier(mode, args).cuda()
    assert all(p.requires_grad for p in model.parameters())
    parameters = sum(p.numel() for p in model.parameters())
    LOG.info('%s source distillation epoch=%d, all trainable=%d; no teacher or alignment Linear', mode, model.source_epoch, parameters)
    optimizer = torch.optim.AdamW(optimizer_groups(model, args), betas=(.9, .999), eps=1e-8)
    micro = args.joint_batch_size if mode == 'joint' else args.batch_size
    accumulation = args.effective_batch_size // micro
    train = Videos(records['train'], root, mode, True, args.seed)
    validation = Videos(records['validation'], root, mode, False, args.seed)
    val_batches = loader(validation, args.val_batch_size, args.workers)
    batch_count = math.ceil(len(train) / micro)
    updates = math.ceil(len(train) / args.effective_batch_size)
    best, history, step, first_epoch, next_batch = None, [], 0, 1, 0
    totals = dict(loss=0., correct=0, count=0)
    resume_rng = None
    last_path, best_path = [out / 'checkpoints' / f'{mode}_{name}.pt' for name in ['last', 'best']]

    def state(epoch, next_epoch, batch_index, full=True):
        value = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch=epoch,
                     next_epoch=next_epoch, next_batch=batch_index, step=step, config=config, mode=mode,
                     best=copy.deepcopy(best), history=copy.deepcopy(history))
        if full:
            value.update(optimizer=optimizer.state_dict(), rng=rng_state(), partial_train_totals=copy.deepcopy(totals))
        return value

    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        assert saved['config'] == config and saved['mode'] == mode
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        first_epoch, next_batch, step = saved['next_epoch'], saved['next_batch'], saved['step']
        history, best, totals, resume_rng = saved['history'], saved['best'], saved['partial_train_totals'], saved['rng']
        # Recover a newer best if interrupted between atomic best and last saves.
        saved_best = torch.load(best_path, map_location='cpu', weights_only=True)
        assert saved_best['config'] == config
        if saved_best['best']['validation']['correct'] > best['validation']['correct']:
            best = saved_best['best']
        del saved, saved_best
        LOG.info('%s resuming epoch%d batch%d optimizer step%d', mode, first_epoch, next_batch, step)
    if not history:
        initial, _ = evaluate(model, val_batches, records['validation'], provenance, classes, args.smoothing)
        best = dict(epoch=0, validation=initial)
        history.append(dict(epoch=0, validation=initial))
        save(state(0, 1, 0, False), best_path)
        save(state(0, 1, 0), last_path)
        history_outputs(out, mode, history)
        LOG.info('%s initial validation=%d/466 %.2f%%', mode, initial['correct'], initial['accuracy'] * 100)
    for epoch in range(first_epoch, args.epochs + 1):
        train.epoch.value = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        batches = loader(train, micro, args.workers, generator=generator, shuffle=True)
        iterator = iter(batches)
        for _ in range(next_batch):
            next(iterator)
        if resume_rng is not None:
            restore_rng(resume_rng)
            resume_rng = None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if next_batch == 0:
            totals = dict(loss=0., correct=0, count=0)
        begun = last_log = time.perf_counter()
        start_count = totals['count']
        factor = schedule_factor(max(0, step - 1), updates * args.epochs, updates * args.warmup_epochs)
        for batch_index in range(next_batch, batch_count):
            _, videos, labels = next(iterator)
            group_start = batch_index // accumulation * accumulation
            window_count = min(args.effective_batch_size, len(train) - group_start * micro)
            factor = schedule_factor(step, updates * args.epochs, updates * args.warmup_epochs)
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * factor
            labels = labels.cuda(non_blocking=True)
            logits = model(videos.cuda(non_blocking=True))
            loss = F.cross_entropy(logits, labels, label_smoothing=args.smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: epoch{epoch} batch{batch_index}')
            (loss * len(labels) / window_count).backward()
            totals['loss'] += float(loss.detach()) * len(labels)
            totals['correct'] += int((logits.argmax(1) == labels).sum())
            totals['count'] += len(labels)
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count:
                norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient norm')
                if step == 0:
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    atomic_json(out / f'gradient_verification_{mode}.json', dict(passed=True, all_parameters_trainable=True,
                                all_parameters_receive_finite_gradients=True, includes_fc_norm=True,
                                no_teacher_or_distillation_linear=True, gradient_norm=float(norm)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.save_every_updates == 0:
                    save(state(epoch, epoch, batch_index + 1), last_path)
            now = time.perf_counter()
            if now - last_log >= 25 or batch_index + 1 == batch_count:
                rate = (totals['count'] - start_count) / max(now - begun, 1e-6)
                LOG.info('%s epoch%d/%d clips%d/%d loss=%.4f train=%.2f%% %.2fvid/s epochETA=%.0fs peak=%.2fGB',
                         mode, epoch, args.epochs, totals['count'], len(train), totals['loss'] / totals['count'],
                         totals['correct'] / totals['count'] * 100, rate, (len(train) - totals['count']) / max(rate, 1e-6),
                         torch.cuda.max_memory_allocated() / 1e9)
                last_log = now
        assert totals['count'] == len(train)
        row = dict(epoch=epoch, train_loss=totals['loss'] / totals['count'], train_accuracy=totals['correct'] / totals['count'],
                   train_samples=totals['count'], lr_vit_last=args.lr * factor, lr_head=args.head_lr * factor,
                   optimizer_steps=step, seconds_training=time.perf_counter() - begun)
        save(state(epoch, epoch, batch_count), last_path)  # resumable pending validation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            metrics, _ = evaluate(model, val_batches, records['validation'], provenance, classes, args.smoothing)
            row['validation'] = metrics
            row['saved_as_best'] = metrics['correct'] > best['validation']['correct']
            if row['saved_as_best']:
                best = dict(epoch=epoch, validation=metrics)
        history.append(row)
        if row.get('saved_as_best'):
            save(state(epoch, epoch + 1, 0, False), best_path)
        totals = dict(loss=0., correct=0, count=0)
        save(state(epoch, epoch + 1, 0), last_path)
        history_outputs(out, mode, history)
        atomic_json(out / 'progress.json', dict(status='training', mode=mode, completed_epoch=epoch, epochs=args.epochs,
                    best=best, last=row, updated_at=time.strftime('%Y-%m-%d %H:%M:%S')))
        LOG.info('EPOCH COMPLETE %s', json.dumps(row))
        next_batch = 0
        del batches, iterator
        gc.collect()
    saved = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    best = saved['best']
    del saved
    metrics, probabilities = evaluate(model, val_batches, records['validation'], provenance, classes, args.smoothing)
    assert metrics['correct'] == best['validation']['correct']
    predictions = []
    for i, record in enumerate(records['validation']):
        predicted = int(probabilities[i].argmax())
        predictions.append(dict(sample_id=record['sample_id'], mode=mode, checkpoint_epoch=best['epoch'],
                    subject=provenance[record['sample_id']]['subject'], label=record['label'], true_class=classes[record['label']],
                    prediction=predicted, answer=classes[predicted], correct=predicted == record['label'],
                    confidence=float(probabilities[i, predicted]), probabilities=probabilities[i].tolist(),
                    available_frames=record['available_frames'], source_frames=provenance[record['sample_id']]['ir_frames'],
                    selected_indices_zero_based=record['selected_indices_zero_based']))
    assert sum(r['correct'] for r in predictions) == metrics['correct'] and len(predictions) == 466
    (out / f'predictions_{mode}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in predictions), encoding='utf8')
    result = dict(best_epoch=best['epoch'], completed_epochs=args.epochs, parameters=parameters,
                  encoder_parameters=21879936, mlp_parameters=parameters - 21879936, validation=metrics,
                  initial_validation=history[0]['validation'], last_validation=history[-1]['validation'])
    atomic_json(out / f'metrics_{mode}.json', result)
    del model, optimizer, val_batches
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/vit')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'results/distill/checkpoints/best.pt')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--check', action='store_true', help='Check paths/data/assets only; no training')
    p.add_argument('--modes', nargs='+', choices=['ir', 'depth', 'joint'], default=['joint'])
    p.add_argument('--epochs', type=int, default=SETTINGS['vit']['epochs'])
    p.add_argument('--val-every', type=int, default=SETTINGS['vit']['val_every'])
    p.add_argument('--warmup-epochs', type=int, default=SETTINGS['vit']['warmup_epochs'])
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--joint-batch-size', type=int, default=2)
    p.add_argument('--effective-batch-size', type=int, default=32, help='paired videos for joint mode, not modality count')
    p.add_argument('--val-batch-size', type=int, default=4)
    p.add_argument('--encoder-batch-size', type=int, default=4, help='individual modality clips per S forward')
    p.add_argument('--lr', type=float, default=SETTINGS['vit']['lr'])
    p.add_argument('--head-lr', type=float, default=SETTINGS['vit']['head_lr'])
    p.add_argument('--layer-decay', type=float, default=.75)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--dropout', type=float, default=.5)
    p.add_argument('--drop-path', type=float, default=.1)
    p.add_argument('--smoothing', type=float, default=.1)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every-updates', type=int, default=50)
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.check:
        result=check_setup('vit',args.data_root)
        result.update(student_checkpoint=str(args.checkpoint),classification_ready=args.checkpoint.is_file(),
            next_step='train_vit.py' if args.checkpoint.is_file() else 'distill_vit.py first')
        print(json.dumps(result));return
    if not args.checkpoint.is_file():
        raise FileNotFoundError('请先运行python -u distill_vit.py；分类训练默认加载results/distill/checkpoints/best.pt')
    if min(args.epochs, args.val_every, args.batch_size, args.joint_batch_size, args.effective_batch_size,
           args.val_batch_size, args.encoder_batch_size, args.hidden, args.lr, args.head_lr, args.grad_clip, args.save_every_updates) <= 0:
        p.error('Invalid positive argument')
    if args.workers < 0 or not 0 <= args.warmup_epochs < args.epochs or args.weight_decay < 0 or not 0 < args.layer_decay <= 1:
        p.error('Invalid workers/warmup/decay')
    if any(not 0 <= x < 1 for x in [args.dropout, args.drop_path, args.smoothing]):
        p.error('Invalid dropout/drop-path/smoothing')
    if any(args.effective_batch_size % b for b in [args.batch_size, args.joint_batch_size]):
        p.error('Effective batch must be divisible by both microbatches')
    if len(set(args.modes)) != len(args.modes):
        p.error('Duplicate modes')
    args.output_dir, args.checkpoint = args.output_dir.resolve(), args.checkpoint.resolve()
    out = args.output_dir
    if out == PROJECT_ROOT / 'results' or not out.is_relative_to(PROJECT_ROOT / 'results'):
        p.error('Use a separate output under this project/results')
    if args.checkpoint.is_relative_to(out):
        p.error('Input checkpoint cannot be inside output')
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Old outputs protected: use --resume or a new --output-dir')
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(sys.stdout),
                        logging.FileHandler(out / 'run.log', mode='a' if args.resume else 'w', encoding='utf8')])
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, records, provenance, classes, protected = inputs(args.data_root,args.seed)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert 'student' in source and 'linear' in source,'请提供本次H→S蒸馏最佳checkpoint'
    source_epoch = source['epoch']
    source_hash = tensor_hash(source['student'])
    del source
    for file in [*Path(__file__).parent.glob('*.py'),ROOT/'upstream/videomaev2/modeling_finetune.py',args.checkpoint]:
        protected[str(file.resolve())] = sha(file)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ['resume', 'output_dir']}
    config.update(experiment='H-to-S token distilled student + random MLP, supervised full fine-tuning', data_root=str(root),
                  pretraining='best student from this project H-to-S distillation',source_distillation_epoch=source_epoch,
                  source_tensor_sha256=source_hash, classes=classes,
                  train_samples=1995, validation_samples=466, frames=16, image_size=224,
                  architecture='shared S -> global token mean -> pretrained fc_norm; concatenate joint -> LN,Linear256,GELU,Dropout,Linear40',
                  discarded='distillation Linear(384,1280); H absent during classification',
                  trainable='all S parameters including fc_norm; all MLP parameters',
                  precision='FP32 parameters, BF16 S forward, FP32 mean/fc_norm/MLP/CE',
                  effective_batch_unit='32 video pairs in joint, 32 single-modality videos in single mode',
                  sampling='same verified16-frame train epoch augmentation, paired crop/flip; fixed validation16 center crop',
                  scheduler=f'{args.warmup_epochs} warmup epochs, cosine to1%; bias/norm no decay; AdamW(.9,.999),eps1e-8',
                  selection='max validation correct, earliest tie; epoch0/every5/final (interval configurable)',
                  torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0), source_fingerprints=protected)
    if args.resume:
        assert json.loads((out / 'config.json').read_text(encoding='utf8')) == config, 'Resume config/source changed'
    else:
        atomic_json(out / 'config.json', config)
        for split in records:
            atomic_json(out / f'{split}_samples.json', records[split])
    LOG.info('Start modes=%s H-to-S best epoch=%d; GPU=%s', args.modes,source_epoch,config['gpu'])
    results = {}
    for mode in args.modes:
        results[mode] = train_mode(mode, args, config, records, root, provenance['validation'], classes)
        atomic_json(out / 'metrics.json', results)
    for path, expected in protected.items():
        assert sha(Path(path)) == expected, f'Protected input changed: {path}'
    result = dict(status='complete', passed=True, results=results, old_sources_unchanged=True,
                  completed_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(out / 'complete.json', result)
    atomic_json(out / 'progress.json', result)
    lines = ['# 蒸馏后的 ViT-S + MLP 全参数分类训练', '',
             f'H→S蒸馏最佳第{source_epoch}轮学生；丢弃Linear，随机初始化MLP，全参数分类训练。', '',
             '|模态|最佳 epoch|正确数/466|准确率|Macro-F1|', '|---|---:|---:|---:|---:|']
    for mode, value in results.items():
        m = value['validation']
        lines.append(f"|{mode}|{value['best_epoch']}|{m['correct']}/466|{m['accuracy']:.2%}|{m['macro_f1']:.4f}|")
    lines += ['', '验证集参与最佳轮选择；不是独立测试集成绩。逐视频预测见 predictions_*.jsonl，曲线见 curves_*.png/svg/csv。',
              '', '完整超参数、源权重和数据哈希见 config.json。原始与蒸馏权重未覆盖。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    LOG.info('COMPLETE %s', json.dumps(results))


if __name__ == '__main__':
    mp.freeze_support()
    main()
