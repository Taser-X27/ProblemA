"""在validation或test上评估单模型，保存供融合使用的40类概率。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from training.common import ROOT,ASSETS,CACHE,CNN_INPUT_MODE,read,save,inventory,sha
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['vit','cnn'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--split',choices=['validation','eval','test'],default='validation',
                        help='eval是validation的别名；test用于固定权重概率融合')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.split=='eval':args.split='validation'
    if args.batch_size<1 or args.workers<0:parser.error('Invalid batch/workers')
    out=args.output_dir or ROOT/'results'/f'eval_{args.model}_{args.split}'
    if out.exists() and any(out.iterdir()):raise FileExistsError('请使用新的--output-dir')
    assert torch.cuda.is_available(),'需要CUDA环境'
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint_sha256=sha(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=saved['config']
    if args.model=='cnn' and saved.get('modality')!=CNN_INPUT_MODE:
        raise ValueError('CNN评估需要四通道IR权重，例如results/cnn_ir/best_model.pt')
    out.mkdir(parents=True,exist_ok=True)
    if args.model=='vit':
        from training.vit_data import inputs,Videos,loader
        from training.vit_engine import Classifier,evaluate
        root,records,provenance,classes,_=inputs(args.data_root,config['seed'],(args.split,))
        settings=SimpleNamespace(**config)
        settings.checkpoint=ROOT/ASSETS['vit']['file']
        mode=saved['mode']
        model=Classifier(mode,settings,load_initialization=False).cuda().eval()
        model.load_state_dict(saved['model'],strict=True)
        rows=records[args.split]
        data=Videos(rows,root,mode,False,config['seed'])
        metrics,q=evaluate(model,loader(data,args.batch_size,args.workers),rows,
            provenance[args.split],classes,config['smoothing'])
        subjects={k:v['subject'] for k,v in provenance[args.split].items()}
        frames={r['sample_id']:r['available_frames'] for r in rows}
    else:
        from training.cnn_cache import verify_records
        from training.cnn_model import Model
        from training.cnn_data import Videos
        from training.cnn_engine import evaluate
        indexed,classes=inventory(args.data_root,(args.split,))
        cache=CACHE if args.split=='validation' else CACHE.with_name(CACHE.name+'_test')
        if not (cache/'complete.json').is_file():
            command='python prepare_cnn.py'+(' --split test' if args.split=='test' else '')
            raise FileNotFoundError(f'请先运行{command}生成裁剪缓存')
        assert read(cache/'complete.json')['passed']
        verify_records(cache/f'{args.split}_records.json',indexed[args.split])
        data=Videos(args.split,cache=cache)
        model=Model(pretrained=False).cuda().eval()
        model.load_state_dict(saved['model'],strict=True)
        metrics,q=evaluate(model,data,args.batch_size,args.workers)
        rows=data.records;subjects={r['sample_id']:r['subject'] for r in rows}
        frames={r['sample_id']:r['frames'] for r in rows}
    if config['classes']!=classes:raise ValueError('checkpoint与数据的类别顺序不一致')
    if sha(args.checkpoint)!=checkpoint_sha256:raise RuntimeError('评估期间checkpoint发生变化')
    with (out/'predictions.jsonl').open('w',encoding='utf8') as stream:
        for row,probabilities in zip(rows,q):
            probabilities=probabilities.tolist()
            pred=max(range(40),key=probabilities.__getitem__)
            result=dict(sample_id=row['sample_id'],subject=subjects[row['sample_id']],label=row['label'],
                true_class=classes[row['label']],prediction=pred,answer=classes[pred],
                correct=pred==row['label'],probabilities=probabilities,split=args.split,
                available_frames=frames[row['sample_id']],checkpoint_epoch=saved['epoch'])
            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
    save(out/'metrics.json',metrics)
    save(out/'config.json',dict(model=args.model,checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_sha256,epoch=saved['epoch'],split=args.split,classes=classes,
        prediction_sha256=sha(out/'predictions.jsonl')))
    print(json.dumps(dict(samples=metrics['samples'],correct=metrics['correct'],accuracy=metrics['accuracy'])))


if __name__=='__main__':
    main()
