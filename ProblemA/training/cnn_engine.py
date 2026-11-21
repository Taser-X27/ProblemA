# 由train_cnn.py启动：原40轮训练算法，ROOT由入口设为当前项目。
"""训练四通道IR人物/手部共享CNN。"""
import argparse
import copy
from datetime import datetime,timezone,timedelta
import gc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

from .common import ROOT,CACHE,CNN_INPUT_MODE,CNN_INPUT_CHANNELS,PRETRAINED,PRETRAINED_SHA,RECIPE,read,save,sha,verify_upstream,inventory,check_setup
from .cnn_cache import verify_records
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from training.cnn_data import Videos,MEAN,STD,epoch_draws
from .cnn_model import Model,optimizer_groups,OriginalJoint
from training.cnn_init import verify_stem

LOG=logging.getLogger('train')


def atomic_checkpoint(path,value):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)

def rng_state():
    a=np.random.get_state()
    return dict(python=random.getstate(),numpy=[a[0],a[1].tolist(),a[2],a[3],a[4]],
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(state):
    random.setstate(state['python']);a=state['numpy']
    np.random.set_state((a[0],np.array(a[1],dtype=np.uint32),a[2],a[3],a[4]))
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def bn_hash(model):
    digest=hashlib.sha256()
    for name,tensor in model.encoder.named_buffers():
        if name.endswith(('running_mean','running_var','num_batches_tracked')):
            digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def loader(dataset,batch,workers,seed):
    return DataLoader(dataset,batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,
                      drop_last=False,generator=torch.Generator().manual_seed(seed))

def score(y,q):
    pred=q.argmax(1)
    cm=np.zeros((40,40),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    support=cm.sum(1);tp=cm.diagonal();predicted=cm.sum(0)
    recall=np.divide(tp,support,out=np.zeros(40),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(40),where=predicted>0)
    f1=np.divide(2*recall*precision,recall+precision,out=np.zeros(40),where=(recall+precision)>0)
    return dict(samples=len(y),correct=int(tp.sum()),accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),
                balanced_accuracy=float(recall.mean()),confusion_matrix=cm.tolist(),
                per_class=[dict(class_id=k,samples=int(support[k]),correct=int(tp[k]),recall=float(recall[k]),
                                precision=float(precision[k]),f1=float(f1[k])) for k in range(40)])

@torch.inference_mode()
def evaluate(model,dataset,batch,workers):
    model.eval()
    probabilities=np.zeros((len(dataset),40),dtype=np.float64)
    labels=np.full(len(dataset),-1,dtype=np.int64)
    begin=time.perf_counter()
    for i,(rows,body,hands,y) in enumerate(loader(dataset,batch,workers,3407)):
        body=body.cuda(non_blocking=True);hands=hands.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):
            a=model(body,hands).float()
            b=model(body.flip(-1),hands.flip(-1)).float()
        q=((a+b)*.5).softmax(1).cpu().numpy()
        # Validation dataset uses canonical row order, not a sampled training subset.
        probabilities[rows.numpy()]=q;labels[rows.numpy()]=y.numpy()
        if i==0 or (i+1)%25==0 or (i+1)*batch>=len(dataset):
            LOG.info('validation %d/%d elapsed=%.1fs',min((i+1)*batch,len(dataset)),len(dataset),time.perf_counter()-begin)
    assert labels.tolist()==[r['label'] for r in dataset.records] and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1),1,atol=1e-6)
    metrics=score(labels,probabilities)
    metrics['loss']=float(-np.log(np.maximum(probabilities[np.arange(len(labels)),labels],1e-12)).mean())
    metrics['loss_smoothed']=float((-(1-RECIPE['smoothing'])*np.log(np.maximum(probabilities[np.arange(len(labels)),labels],1e-12))
                                   -RECIPE['smoothing']*np.log(np.maximum(probabilities,1e-12)).mean(1)).mean())
    metrics['per_subject']={}
    for subject in sorted({r['subject'] for r in dataset.records}):
        mask=np.array([r['subject']==subject for r in dataset.records])
        metrics['per_subject'][subject]=dict(samples=int(mask.sum()),correct=int((probabilities[mask].argmax(1)==labels[mask]).sum()),
            accuracy=float((probabilities[mask].argmax(1)==labels[mask]).mean()))
    return metrics,probabilities

def prediction_log(path,dataset,classes,probabilities,epoch):
    with path.open('w',encoding='utf8') as stream:
        for r,q in zip(dataset.records,probabilities,strict=True):
            pred=int(q.argmax())
            row=dict(sample_id=r['sample_id'],subject=r['subject'],source_clip=r['source_clip'],label=r['label'],
                true_class=classes[r['label']],prediction=pred,answer=classes[pred],correct=pred==r['label'],
                probabilities=q.tolist(),available_frames=r['frames'],checkpoint_epoch=epoch)
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')

def curve_outputs(out,history):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    if Path('C:/Windows/Fonts/msyh.ttc').exists():
        fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'none'})
    save(out/'history.json',history)
    (out/'history.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in history),encoding='utf8')
    train=[r for r in history if 'train_loss' in r];val=[r for r in history if 'validation' in r]
    fields=['epoch','train_loss','train_fused_loss','train_accuracy','validation_loss','validation_accuracy']
    with (out/'training_curves.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in history:
            writer.writerow({k:row.get(k) for k in fields[:4]}|
                dict(validation_loss=row.get('validation',{}).get('loss_smoothed'),
                     validation_accuracy=row.get('validation',{}).get('accuracy')))
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    axes[0].plot([r['epoch'] for r in train],[r['train_loss'] for r in train],label='训练总损失（含两路辅助项）',alpha=.65)
    axes[0].plot([r['epoch'] for r in train],[r['train_fused_loss'] for r in train],label='训练融合头损失（平滑0.05）')
    axes[0].plot([r['epoch'] for r in val],[r['validation']['loss_smoothed'] for r in val],'o-',label='验证融合输出损失（平滑0.05，含TTA）')
    axes[1].plot([r['epoch'] for r in train],[100*r['train_accuracy'] for r in train],label='训练（随机增强＋类别均衡采样）')
    axes[1].plot([r['epoch'] for r in val],[100*r['validation']['accuracy'] for r in val],'o-',label='验证（原图＋翻转）')
    for ax,ylabel in zip(axes,['Loss','准确率（%）']):
        ax.set(xlabel='Epoch',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1].set_ylim(0,103)
    fig.suptitle('CNN 四通道IR人物＋手部共享R(2+1)D-34＋交互Transformer；YOLO26n-pose')
    for ext in ['png','svg']:
        fig.savefig(out/f'training_curves.{ext}',dpi=160)
    plt.close(fig)

def factor(step,total,warmup):
    if step<warmup:
        return (step+1)/warmup
    t=min(1.,max(0.,(step-warmup)/max(1,total-warmup-1)))
    return RECIPE['min_lr_factor']+(1-RECIPE['min_lr_factor'])*.5*(1+math.cos(math.pi*t))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/cnn_ir')
    parser.add_argument('--batch-size',type=int,default=RECIPE['batch_size'])
    parser.add_argument('--grad-accum',type=int,default=RECIPE['grad_accum'])
    parser.add_argument('--val-batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        print(json.dumps(check_setup('cnn',args.data_root)));return
    assert args.batch_size*args.grad_accum==16,'Preserve author effective batch16'
    out=args.output_dir.resolve()
    assert out.is_relative_to(ROOT/'results') and out!=ROOT/'results'
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError('Old outputs protected: use --resume or another output directory')
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',force=True,handlers=[
        logging.StreamHandler(),logging.FileHandler(out/'train.log',mode='a' if args.resume else 'w',encoding='utf8')])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    seed=RECIPE['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    assert read(CACHE/'complete.json')['passed'] and sha(PRETRAINED)==PRETRAINED_SHA
    indexed,classes=inventory(args.data_root)
    for name in ['train','validation']:
        verify_records(CACHE/f'{name}_records.json',indexed[name])
    ntrain=len(indexed['train']);nval=len(indexed['validation'])
    source_files=[*Path(__file__).parent.glob('*.py'),ROOT/'configs/dataset.json',ROOT/'configs/training.json',ROOT/'configs/assets.json',CACHE/'identity.json',CACHE/'complete.json',CACHE/'train_records.json',CACHE/'validation_records.json',PRETRAINED]
    protected={str(p):sha(p) for p in source_files}
    config=dict(recipe=RECIPE,batch_size=args.batch_size,grad_accum=args.grad_accum,val_batch_size=args.val_batch_size,
        workers=args.workers,activation_checkpointing=True,train_samples=ntrain,validation_samples=nval,classes=classes,
        mode='IR only; body and hands each replicate IR to four channels',
        input_mode=CNN_INPUT_MODE,channel_order=CNN_INPUT_CHANNELS,
        normalization=dict(mean=MEAN.flatten().tolist(),std=STD.flatten().tolist()),
        input_source='IR channel from the YOLO26n / YOLO26n-pose crop cache',
        pretrained_stem='first3 = original RGB weights; fourth = RGB weight mean',
        pretraining='IG65M -> K400, no CUHK classifier initialization',
        implementation='author inference architecture; training loop reconstructed from technical report Table8',
        reconstruction_choices=['inverse-frequency sampling with replacement; 1995 draws/epoch',
          'linear LR warmup from zero, cosine per optimizer update',
          'zoom scale interpreted as side fraction; independent deterministic RNG per clip/view',
          'AdamW default betas0.9,0.999 eps1e-8; all parameters weight-decayed'],
        differences=['YOLO26n person detector and YOLO26n-pose; new project-local data and cache',
          'local1995 train/466 validation rather than author3036 training/405 test',
          'native local304x228 images retained; author hand min_side96 pixels unchanged',
          'one shared IR body/hands member, without other ensemble members or structure decoder',
          'activation checkpointing for12GB GPU; effective batch16 preserved',
          'validation every5epochs, best epoch selected on local validation; final40 separately retained'],
        protected_sources=protected,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
        upstream=verify_upstream())
    if (out/'config.json').exists():
        assert read(out/'config.json')==config,'Training identity changed'
    else:
        save(out/'config.json',config)
    if (out/'complete.json').exists():
        LOG.info('Training already complete');return
    model=Model(pretrained=True,activation_checkpointing=True)
    if not args.resume:
        save(out/'pretraining_verification.json',verify_stem(model,PRETRAINED))
    model=model.cuda()
    model.requires_grad_(True)
    LOG.info('IR x 4 joint model parameters=%d; encoder shared, removed temporal strides=%s',sum(p.numel() for p in model.parameters()),model.removed)
    optimizer=torch.optim.AdamW(optimizer_groups(model,RECIPE),weight_decay=RECIPE['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=RECIPE['grad_scaler_init'])
    history=[];best_correct=-1;best_epoch=-1;step=0;first_epoch=1;next_batch=0;partial=None
    initial_bn_hash=bn_hash(model)
    resume_rng=None
    if args.resume and (out/'last_checkpoint.pt').exists():
        saved=torch.load(out/'last_checkpoint.pt',map_location='cpu',weights_only=True)
        assert saved['config']==config
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];best_correct=saved['best_correct'];best_epoch=saved['best_epoch'];step=saved['optimizer_steps']
        first_epoch=saved['epoch']+1 if saved['epoch_complete'] else saved['epoch']
        next_batch=0 if saved['epoch_complete'] else saved['next_batch']
        partial=None if saved['epoch_complete'] else saved['partial_totals']
        initial_bn_hash=saved['initial_bn_hash'];resume_rng=saved['rng']
        del saved
        LOG.info('Restored epoch%d batch%d optimizer_step%d',first_epoch,next_batch,step)
    train_records=read(CACHE/'train_records.json');val=Videos('validation')
    batches_per_epoch=math.ceil(ntrain/args.batch_size)
    steps_per_epoch=math.ceil(batches_per_epoch/args.grad_accum)
    total_steps=RECIPE['epochs']*steps_per_epoch;warmup=RECIPE['warmup_epochs']*steps_per_epoch
    def checkpoint_state(epoch,cursor,complete,totals):
        atomic_checkpoint(out/'last_checkpoint.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),config=config,epoch=epoch,next_batch=cursor,epoch_complete=complete,
            partial_totals=totals,history=history,best_correct=best_correct,best_epoch=best_epoch,optimizer_steps=step,
            initial_bn_hash=initial_bn_hash,rng=rng_state(),modality=CNN_INPUT_MODE))
    if not history:
        initial,q=evaluate(model,val,args.val_batch_size,args.workers)
        history.append(dict(epoch=0,validation=initial))
        save(out/'metrics_initial.json',initial)
        curve_outputs(out,history)
        checkpoint_state(1,0,False,None)
        LOG.info('Initial validation %d/466 %.2f%%',initial['correct'],initial['accuracy']*100)
    if resume_rng is not None:
        restore_rng(resume_rng)
    job=dict(pid=os.getpid(),status='running',started=datetime.now(timezone(timedelta(hours=8))).isoformat(),
             output=str(out),command=__import__('sys').argv)
    save(out/'job.json',job)
    try:
        for epoch in range(first_epoch,RECIPE['epochs']+1):
            begin=time.perf_counter();model.train()
            assert not any(m.training for m in model.encoder.modules() if isinstance(m,nn.BatchNorm3d))
            drawn=epoch_draws(train_records,epoch)
            dataset=Videos('train',True,epoch,drawn)
            batches=loader(dataset,args.batch_size,args.workers,RECIPE['seed']+epoch)
            totals=partial or dict(loss=0.,fused=0.,correct=0,samples=0,skipped_updates=0)
            optimizer.zero_grad(set_to_none=True)
            for batch_index,(_,body,hands,y) in enumerate(batches):
                if batch_index<next_batch:
                    continue
                group_start=(batch_index//args.grad_accum)*args.grad_accum
                group_samples=min(ntrain,(group_start+args.grad_accum)*args.batch_size)-group_start*args.batch_size
                if batch_index%args.grad_accum==0:
                    multiplier=factor(step,total_steps,warmup)
                    for group in optimizer.param_groups:group['lr']=group['base_lr']*multiplier
                body=body.cuda(non_blocking=True);hands=hands.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    fused,body_head,hand_head=model(body,hands,True)
                    main_loss=F.cross_entropy(fused.float(),y,label_smoothing=RECIPE['smoothing'])
                    loss=main_loss+RECIPE['aux_weight']*(F.cross_entropy(body_head.float(),y,label_smoothing=RECIPE['smoothing'])+
                                        F.cross_entropy(hand_head.float(),y,label_smoothing=RECIPE['smoothing']))
                assert torch.isfinite(loss)
                scaler.scale(loss*(len(y)/group_samples)).backward()
                totals['loss']+=float(loss.detach())*len(y);totals['fused']+=float(main_loss.detach())*len(y)
                totals['correct']+=int((fused.argmax(1)==y).sum());totals['samples']+=len(y)
                del body,hands,y,fused,body_head,hand_head,loss,main_loss
                if (batch_index+1)%args.grad_accum==0 or batch_index+1==batches_per_epoch:
                    scaler.unscale_(optimizer)
                    norm=nn.utils.clip_grad_norm_(model.parameters(),RECIPE['grad_clip'])
                    scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    totals['skipped_updates']+=int(scaler.get_scale()<scale_before)
                    optimizer.zero_grad(set_to_none=True);step+=1
                    if step%25==0 or batch_index+1==batches_per_epoch:
                        elapsed=time.perf_counter()-begin
                        LOG.info('epoch %d/%d clips %d/%d loss=%.4f train_ac=%.2f%% lr=%s elapsed=%.1fs ETA_epoch=%.1fs',
                            epoch,RECIPE['epochs'],totals['samples'],ntrain,totals['loss']/totals['samples'],100*totals['correct']/totals['samples'],
                            [g['lr'] for g in optimizer.param_groups],elapsed,
                            elapsed*max(0,ntrain-totals['samples'])/max(1,totals['samples']-(partial or {}).get('samples',0)))
                        checkpoint_state(epoch,batch_index+1,False,totals)
            assert totals['samples']==ntrain and initial_bn_hash==bn_hash(model)
            row=dict(epoch=epoch,train_loss=totals['loss']/ntrain,train_fused_loss=totals['fused']/ntrain,
                train_accuracy=totals['correct']/ntrain,train_draws=ntrain,unique_train_clips=len(set(drawn)),
                skipped_updates=totals['skipped_updates'],seconds_training=time.perf_counter()-begin,
                optimizer_steps=step,learning_rates=[g['lr'] for g in optimizer.param_groups])
            if epoch%RECIPE['validation_every']==0 or epoch==RECIPE['epochs']:
                gc.collect();torch.cuda.empty_cache()
                metrics,q=evaluate(model,val,args.val_batch_size,args.workers)
                row['validation']=metrics
                if metrics['correct']>best_correct:
                    best_correct=metrics['correct'];best_epoch=epoch
                    atomic_checkpoint(out/'best_model.pt',dict(model=model.state_dict(),epoch=epoch,metrics=metrics,
                                      config=config,strideless=True,modality=CNN_INPUT_MODE))
                    save(out/'metrics_best.json',dict(epoch=epoch,**metrics))
                    prediction_log(out/'predictions_best.jsonl',val,classes,q,epoch)
                LOG.info('epoch%d validation %d/466 %.2f%%; best epoch%d %d/466',epoch,metrics['correct'],metrics['accuracy']*100,best_epoch,best_correct)
            history.append(row);curve_outputs(out,history)
            checkpoint_state(epoch,batches_per_epoch,True,totals)
            save(out/'progress.json',dict(epoch=epoch,total_epochs=RECIPE['epochs'],best_epoch=best_epoch,best_correct=best_correct,
                latest=row,optimizer_steps=step,BN_statistics_unchanged=True))
            next_batch=0;partial=None
        atomic_checkpoint(out/'last_model.pt',dict(model=model.state_dict(),epoch=RECIPE['epochs'],config=config,strideless=True,modality=CNN_INPUT_MODE))
        prediction_log(out/'predictions_last.jsonl',val,classes,q,RECIPE['epochs'])
        save(out/'metrics_last.json',dict(epoch=RECIPE['epochs'],**history[-1]['validation']))
        for path,digest in protected.items():assert sha(path)==digest
        save(out/'complete.json',dict(passed=True,epochs=RECIPE['epochs'],best_epoch=best_epoch,best_correct=best_correct,
             best_accuracy=best_correct/nval,BN_statistics_unchanged=True,original_weights_unchanged=True))
        job['status']='complete';save(out/'job.json',job)
        (out/'report.md').write_text(f'# CNN 四通道IR人物＋手部训练结果\n\n最佳第{best_epoch}轮：{best_correct}/466，{best_correct/nval:.2%}。\n\n'
            '训练仅使用1995条train，466条validation用于每5轮验证和最佳轮选择。\n\n'
            '![训练曲线](training_curves.png)\n\n配方和全部复现边界见config.json与项目README.md。\n',encoding='utf8')
        LOG.info('COMPLETE best epoch%d %d/466 %.2f%%',best_epoch,best_correct,100*best_correct/nval)
    except BaseException as error:
        job['status']='interrupted';job['error']=repr(error);save(out/'job.json',job)
        LOG.exception('Interrupted; resume from the last saved optimizer boundary')
        raise


if __name__=='__main__':
    mp.freeze_support()
    main()
