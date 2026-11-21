"""从裁剪缓存读取单通道IR，为CNN生成四通道IR输入。"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .common import CACHE,RECIPE
from .cnn_input import CHANNEL_ORDER

MEAN=torch.tensor([.43216,.394666,.37645,(.43216+.394666+.37645)/3]).view(4,1,1,1)
STD=torch.tensor([.22803,.22145,.216989,(.22803+.22145+.216989)/3]).view(4,1,1,1)
IR_CACHE_CHANNEL=CHANNEL_ORDER.index('IR')


def augment(x,rng):
    # 与原CNN训练增强一致；所有帧共享同一变换。
    if rng.random()<.5:x=x.flip(-1)
    if rng.random()<.5:x=x.roll(int(rng.integers(-1,2)),dims=1)
    if rng.random()<.5:
        side=int(round(128*rng.uniform(.75,1.)))
        top=int(rng.integers(0,129-side));left=int(rng.integers(0,129-side))
        x=F.interpolate(x.permute(1,0,2,3)[:,:,top:top+side,left:left+side],size=(128,128),
            mode='bilinear',align_corners=False).permute(1,0,2,3)
    x=(x*float(rng.uniform(.85,1.15))).clamp_(0,1)
    if rng.random()<.25:
        h=int(round(128*rng.uniform(.1,.3)));w=int(round(128*rng.uniform(.1,.3)))
        top=int(rng.integers(0,129-h));left=int(rng.integers(0,129-w))
        x[:,:,top:top+h,left:left+w]=0
    return x


class Videos(Dataset):
    def __init__(self,split,training=False,epoch=0,selected=None,cache=None):
        self.cache=Path(cache or CACHE).resolve()
        identity=json.loads((self.cache/'identity.json').read_text(encoding='utf8'))
        if identity.get('input_mode')!='depth_ir' or identity.get('channel_order', CHANNEL_ORDER)!=CHANNEL_ORDER:
            raise ValueError('裁剪缓存应按[Depth_R,Depth_G,Depth_B,IR]存储')
        self.split,self.training,self.epoch=split,training,epoch
        self.records=json.loads((self.cache/f'{split}_records.json').read_text(encoding='utf8'))
        self.selected=list(range(len(self.records))) if selected is None else list(selected)
        self._open_arrays()

    def _open_arrays(self):
        # 工作进程使用序列化保留的self.cache，不重新读取旧工程全局CACHE。
        self.body=np.load(self.cache/f'{self.split}_body.npy',mmap_mode='r')
        self.hands=np.load(self.cache/f'{self.split}_hands.npy',mmap_mode='r')
        assert self.body.shape==self.hands.shape==(len(self.records),32,len(CHANNEL_ORDER),128,128)

    def __len__(self):return len(self.selected)

    def __getstate__(self):
        state=self.__dict__.copy();state.pop('body',None);state.pop('hands',None)
        return state

    def __setstate__(self,state):
        self.__dict__.update(state);self._open_arrays()

    def __getitem__(self,index):
        row=self.selected[index];views=[]
        for view,source in enumerate([self.body,self.hands]):
            ir=source[row,:,IR_CACHE_CHANNEL:IR_CACHE_CHANNEL+1].copy()
            x=torch.from_numpy(ir).permute(1,0,2,3).float().div_(255)
            if self.training:
                x=augment(x,np.random.default_rng(np.random.SeedSequence([RECIPE['seed'],self.epoch,row,view])))
            # 匹配Noah的IR分支：先扩展灰度，再用四组Kinetics统计量归一化。
            views.append(((x.expand(4,-1,-1,-1)-MEAN)/STD).contiguous())
        return row,*views,self.records[row]['label']


def epoch_draws(records,epoch):
    labels=np.array([r['label'] for r in records]);counts=np.bincount(labels,minlength=40)
    weights=1/counts[labels];weights=weights/weights.sum()
    return np.random.default_rng(np.random.SeedSequence([RECIPE['seed'],epoch,917])).choice(
        len(records),size=len(records),replace=True,p=weights).tolist()
