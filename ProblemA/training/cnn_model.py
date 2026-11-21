"""Author-compatible shared encoder, auxiliary heads and cross-view interaction."""
from .common import PRETRAINED, PRETRAINED_SHA, sha, verify_upstream
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

verify_upstream()
from sota_joint import JointModel as OriginalJoint
from sota_model import Model as OriginalModel
import sota_model


class Model(OriginalJoint):
    def __init__(self,pretrained=True,activation_checkpointing=True):
        super().__init__(strided=False,interaction=True,frames=32)
        self.activation_checkpointing=activation_checkpointing
        assert len(self.removed)==6,self.removed
        if pretrained:
            assert sha(PRETRAINED)==PRETRAINED_SHA
            sota_model.WEIGHTS=PRETRAINED
            source=OriginalModel(pretrained=True)
            self.encoder.load_state_dict(source.encoder.state_dict(),strict=True)
            del source
        # Report section4.3: interaction logits start at zero.
        nn.init.zeros_(self.interaction.out.weight)
        nn.init.zeros_(self.interaction.out.bias)
        assert sum(p.numel() for p in self.interaction.parameters())==1206568
        self.train()

    def train(self,mode=True):
        super().train(mode)
        # Freeze running statistics, not the learnable BN affine parameters.
        for module in self.encoder.modules():
            if isinstance(module,nn.BatchNorm3d):
                module.eval()
        return self

    def forward(self,body,hands,auxiliary=False):
        n=len(body)
        assert body.shape==hands.shape and body.shape[1:]==(4,32,128,128)
        x=torch.cat([body,hands])
        stages=[self.encoder.stem,*self.encoder.layer1,*self.encoder.layer2,*self.encoder.layer3,*self.encoder.layer4]
        for stage in stages:
            if self.training and self.activation_checkpointing:
                x=checkpoint(stage,x,use_reentrant=False,preserve_rng_state=True)
            else:
                x=stage(x)
        assert x.shape[2]==32,x.shape
        z=x.mean((-1,-2,-3))
        fused=self.fusion(torch.cat([z[:n],z[n:]],1))+self.interaction(x[:n],x[n:])
        if auxiliary:
            return fused,self.head(z[:n]),self.hand_head(z[n:])
        return fused

def optimizer_groups(model,recipe):
    return [dict(params=list(model.encoder.parameters()),lr=recipe['lr_encoder'],base_lr=recipe['lr_encoder']),
            dict(params=[*model.head.parameters(),*model.hand_head.parameters(),*model.fusion.parameters()],
                 lr=recipe['lr_head'],base_lr=recipe['lr_head']),
            dict(params=list(model.interaction.parameters()),lr=recipe['lr_interaction'],base_lr=recipe['lr_interaction'])]
