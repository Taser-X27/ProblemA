"""项目内VideoMAEv2 H/S骨干；H用于冻结教师，S用于蒸馏和分类。"""
import hashlib
import importlib.util
import sys
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from .common import ROOT,ASSETS,sha
from unittest.mock import patch


def attention_sdpa(self,x):
    b,n,c=x.shape
    bias=None if self.q_bias is None else torch.cat([self.q_bias,torch.zeros_like(self.v_bias),self.v_bias])
    qkv=F.linear(x,self.qkv.weight,bias).reshape(b,n,3,self.num_heads,-1)
    q,k,v=qkv.permute(2,0,3,1,4).unbind(0)
    x=F.scaled_dot_product_attention(q,k,v,scale=self.scale,dropout_p=self.attn_drop.p if self.training else 0.)
    return self.proj_drop(self.proj(x.transpose(1,2).reshape(b,n,c)))


def official():
    name='project_videomaev2'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'upstream/videomaev2/modeling_finetune.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Overwriting vit_.*')
            spec.loader.exec_module(module)
        module.Attention.forward=attention_sdpa
    return sys.modules[name]


def build_encoder(kind='small',checkpoint=None):
    if kind=='huge':
        return build_teacher()
    assert kind=='small'
    path=ROOT/ASSETS['vit']['file'] if checkpoint is None else checkpoint
    assert sha(path)==ASSETS['vit']['sha256']
    model=official().vit_small_patch16_224(num_classes=710,all_frames=16,tubelet_size=2,
        drop_path_rate=0.,use_mean_pooling=True)
    model.head=nn.Identity()
    source=torch.load(path,map_location='cpu',weights_only=True)
    state={k:v for k,v in source['module'].items() if not k.startswith('head.')}
    model.load_state_dict(state,strict=True)
    del model.pos_embed
    model.register_buffer('pos_embed',official().get_sinusoid_encoding_table(1568,384),persistent=False)
    model.float()
    assert sum(p.numel() for p in model.parameters())==21879936
    return model,dict(source='official K710 giant-distilled ViT-S',strict_load=True)


def build_teacher():
    """meta构建避免额外初始化2.53GB随机权重；只读mmap加载公开H骨干。"""
    path=ROOT/ASSETS['teacher']['file']
    assert sha(path)==ASSETS['teacher']['sha256']
    linspace=torch.linspace
    def cpu_linspace(*args,**kwargs):
        kwargs['device']='cpu'
        return linspace(*args,**kwargs)
    with patch.object(torch,'linspace',cpu_linspace),torch.device('meta'):
        model=official().vit_huge_patch16_224(num_classes=101,all_frames=16,tubelet_size=2,
            drop_path_rate=0.,use_mean_pooling=True)
    model.head=nn.Identity()
    source=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(source['model'],strict=True,assign=True)
    del model.pos_embed
    model.register_buffer('pos_embed',official().get_sinusoid_encoding_table(1568,1280),persistent=False)
    model.float().eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters())==631607040
    assert not any(p.is_meta or p.requires_grad for p in model.parameters())
    return model,dict(source='official mae-h UCF101 fine-tuned backbone',parameters=631607040,
        token_shape=[1568,1280],strict_load=True)


def forward_tokens(model,x):
    x=model.patch_embed(x)
    x=model.pos_drop(x+model.pos_embed.to(device=x.device,dtype=x.dtype))
    for block in model.blocks:x=block(x)
    return x


def tensor_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
