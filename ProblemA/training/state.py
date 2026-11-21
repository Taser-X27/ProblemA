"""原子保存、随机状态恢复及ViT逐block前向。"""
import random
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def save(obj,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pt.tmp')
    torch.save(obj,temp);temp.replace(path)


def rng_state():
    a=np.random.get_state()
    return dict(python=random.getstate(),numpy=[a[0],a[1].tolist(),a[2],a[3],a[4]],
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);a=state['numpy']
    np.random.set_state((a[0],np.array(a[1],dtype=np.uint32),a[2],a[3],a[4]))
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def student_tokens(model,x,use_checkpoint=True):
    x=model.patch_embed(x)
    x=model.pos_drop(x+model.pos_embed.to(device=x.device,dtype=x.dtype))
    for block in model.blocks:
        x=checkpoint(block,x,use_reentrant=False) if use_checkpoint and model.training else block(x)
    return x
