"""检查新工程；可选模型前向，无优化器、无反向传播、不启动训练。"""
import argparse
import gc
import json
import logging
from pathlib import Path
import tempfile
import sys
from training.common import ROOT,ASSETS,check_setup,data_root,inventory,read,save
from training.cnn_input import decode_depth_ir
from training.cnn_init import verify_stem
from training.cnn_data import Videos as CNNVideos,MEAN as CNN_MEAN,STD as CNN_STD
from training.cnn_model import Model
from training.vit_data import inputs,Videos as VitVideos
from training.vit_engine import Classifier
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward',action='store_true',help='额外执行两模型CUDA前向，不训练')
    parser.add_argument('--distill-forward',action='store_true',help='额外检查H→S对齐前向，不反向传播')
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=check_setup()
    indexed,_=inventory()
    record=indexed['train'][0]
    windows=dict(body=[.1,.05,.9,.95],hands=[.25,.25,.75,.75])
    arrays=decode_depth_ir(record,windows,data_root())
    assert all(a.shape==(32,4,128,128) for a in arrays)
    (ROOT/'cache').mkdir(exist_ok=True)
    # 以两段真实视频检查完整缓存写入与复用；预设窗口，不运行检测或训练。
    from training import cnn_prepare
    old_cache,old_inventory=cnn_prepare.CACHE,cnn_prepare.inventory
    old_argv=sys.argv
    old_handlers=list(logging.getLogger().handlers)
    try:
        with tempfile.TemporaryDirectory(prefix='prepare_check_',dir=ROOT/'cache') as folder:
            folder=Path(folder)
            small={name:[rows[0]] for name,rows in indexed.items()}
            cnn_prepare.CACHE=folder
            cnn_prepare.inventory=lambda *a,**kw:(small,read(ROOT/'dataset/train/manifest.json')['classes'])
            save(folder/'windows.json',{r['sample_id']:{**windows,'status':'check'} for rows in small.values() for r in rows})
            sys.argv=['prepare_cnn.py','--workers','2']
            try:
                cnn_prepare.main()
                cnn_prepare.main()
                assert read(folder/'complete.json')['passed']
                for name in small:
                    assert np.load(folder/f'{name}_body.npy').shape==(1,32,4,128,128)
            finally:
                for handler in list(logging.getLogger().handlers):
                    if handler not in old_handlers:
                        logging.getLogger().removeHandler(handler);handler.close()
    finally:
        cnn_prepare.CACHE,cnn_prepare.inventory=old_cache,old_inventory
        sys.argv=old_argv
    with tempfile.TemporaryDirectory(prefix='input_check_',dir=ROOT/'cache') as folder:
        folder=Path(folder)
        save(folder/'identity.json',dict(input_mode='depth_ir'))
        save(folder/'train_records.json',[record])
        for kind,array in zip(['body','hands'],arrays):np.save(folder/f'train_{kind}.npy',array[None])
        with_ir=CNNVideos('train',cache=folder)
        try:
            for array,tensor in zip(arrays,with_ir[0][1:3]):
                ir=torch.from_numpy(array[:,3].copy()).float().div_(255).unsqueeze(0)
                expected=(ir.repeat(4,1,1,1)-CNN_MEAN)/CNN_STD
                torch.testing.assert_close(tensor,expected,rtol=0,atol=0)
        finally:
            with_ir.body._mmap.close();with_ir.hands._mmap.close()
        dataset=CNNVideos('train',training=True,epoch=1,cache=folder)
        direct=dataset[0]
        for _,body,hands,labels in DataLoader(dataset,batch_size=1,num_workers=2):
            torch.testing.assert_close(body[0],direct[1],rtol=0,atol=0)
            torch.testing.assert_close(hands[0],direct[2],rtol=0,atol=0)
        dataset.body._mmap.close();dataset.hands._mmap.close()
    cnn=Model(pretrained=True).eval().requires_grad_(False)
    result['cnn_initialization']=verify_stem(cnn,ROOT/ASSETS['cnn']['file'])
    result['cnn_input_ir_replication_verified']=True
    result['Windows_workers2_inputs_identical']=True
    if args.forward:
        assert torch.cuda.is_available()
        cnn=cnn.cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            logits=cnn(direct[1][None].cuda(),direct[2][None].cuda())
        assert logits.shape==(1,40) and torch.isfinite(logits).all()
        result['cnn_forward_shape']=list(logits.shape)
        del logits
    del cnn;gc.collect();torch.cuda.empty_cache()
    root,records,_,_,_=inputs()
    data=VitVideos(records['train'],root,'joint',True,42)
    x=data[0][1][None]
    assert x.shape==(1,2,3,16,224,224)
    settings=SimpleNamespace(no_gradient_checkpointing=False,encoder_batch_size=2,
        checkpoint=ROOT/ASSETS['vit']['file'],drop_path=.1,hidden=256,dropout=.5)
    vit=Classifier('joint',settings,load_initialization=False).eval().requires_grad_(False)
    result['vit_parameters']=sum(p.numel() for p in vit.parameters())
    assert result['vit_parameters']==22088616
    # 仅构造一个临时checkpoint，检查阶段二确实加载student并丢弃Linear。
    with tempfile.TemporaryDirectory(prefix='student_load_check_',dir=ROOT/'cache') as folder:
        path=Path(folder)/'student.pt'
        torch.save(dict(student=vit.backbone.state_dict(),linear=torch.nn.Linear(384,1280).state_dict(),
            epoch=0,best=dict(epoch=0),config=dict(aligned_shape=[1568,1280])),path)
        settings.checkpoint=path
        restored=Classifier('joint',settings).eval().requires_grad_(False)
        for key,value in vit.backbone.state_dict().items():
            torch.testing.assert_close(value,restored.backbone.state_dict()[key],rtol=0,atol=0)
        assert restored.source_epoch==0
        assert not any(key.startswith('linear.') for key in restored.state_dict())
        del restored
    result['stage2_student_checkpoint_loader_verified']=True
    if args.forward:
        with torch.inference_mode():logits=vit.cuda()(x.cuda())
        assert logits.shape==(1,40) and torch.isfinite(logits).all()
        result['vit_forward_shape']=list(logits.shape)
    if args.distill_forward:
        from training.vit_encoder import build_encoder
        from training.distill_engine import alignment
        assert torch.cuda.is_available()
        del vit;gc.collect();torch.cuda.empty_cache()
        teacher,_=build_encoder('huge')
        teacher=teacher.cuda().eval().requires_grad_(False)
        student,_=build_encoder('small')
        student=student.cuda().eval().requires_grad_(False)
        linear=torch.nn.Linear(384,1280).cuda().eval().requires_grad_(False)
        with torch.inference_mode():
            mse,cosine,energy=alignment(student,linear,teacher,x[:,0].cuda(),False)
        assert torch.isfinite(mse).all() and torch.isfinite(cosine).all()
        assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
        result['distillation_forward']=dict(teacher=[1,1568,1280],student=[1,1568,384],
            aligned=[1,1568,1280],finite=True,teacher_frozen=True,optimizer_steps=0)
        del teacher,student,linear;gc.collect();torch.cuda.empty_cache()
    result.update(project_local=True,old_classifier_weights_loaded=False,optimizer_steps=0,training_started=False)
    result['small_cache_preparation_and_reuse_verified']=True
    # 两个训练循环都依赖绘图；检查实际打包的matplotlib可用。
    from training.vit_engine import history_outputs
    from training.cnn_engine import curve_outputs
    with tempfile.TemporaryDirectory(prefix='plot_check_',dir=ROOT/'cache') as folder:
        folder=Path(folder)
        history=[dict(epoch=0,validation=dict(accuracy=0.,loss_smoothed=4.))]
        history_outputs(folder,'joint',history)
        curve_outputs(folder,history)
        assert (folder/'curves_joint.png').is_file() and (folder/'training_curves.png').is_file()
    result['plotting_dependencies_verified']=True
    loaded={}
    for name,module in list(sys.modules.items()):
        if name.startswith('training.') or name.startswith('sota_') or name=='project_videomaev2':
            path=getattr(module,'__file__',None)
            if path:
                path=Path(path).resolve()
                assert path.is_relative_to(ROOT),f'项目外代码依赖：{name} -> {path}'
                loaded[name]=path.relative_to(ROOT).as_posix()
    result['local_implementation_modules']=loaded
    save(ROOT/'results/setup_check.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
