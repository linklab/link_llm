"""One-at-a-time RoPE / RMSNorm / SwiGLU / GQA ablations, including packed/cache paths."""
import importlib.util
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('v045_previous',HERE.parents[1]/'v0.4.4/0.model/lm.py')
previous=importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
module,corpus,packing=previous.module,previous.corpus,previous.packing
ByteBPE,sequence_loss=previous.ByteBPE,previous.sequence_loss
VARIANTS=('baseline','rope','rmsnorm','swiglu','gqa')


class RMSNorm(nn.Module):
    def __init__(self,width,eps=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(width)); self.eps=eps

    def forward(self,x):
        normalized=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
        return normalized.to(x.dtype)*self.weight


class SwiGLU(nn.Module):
    def __init__(self,width,hidden,layers):
        super().__init__()
        self.gate=nn.Linear(width,hidden,bias=False)
        self.up=nn.Linear(width,hidden,bias=False)
        self.down=nn.Linear(hidden,width,bias=False)
        for layer in (self.gate,self.up): nn.init.normal_(layer.weight,std=.02)
        nn.init.normal_(self.down.weight,std=.02/math.sqrt(2*layers))

    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))


class ModernNetwork(previous.CachedNetwork):
    def __init__(self,vocab,embed,context,heads,ffn,dropout,layers,variant='baseline',kv_heads=2):
        if variant not in VARIANTS:
            raise ValueError('unknown architecture variant')
        if variant=='gqa' and (kv_heads<1 or heads%kv_heads):
            raise ValueError('query heads must be divisible by KV heads')
        if variant=='rope' and (embed//heads)%2:
            raise ValueError('RoPE requires an even head dimension')
        super().__init__(vocab,embed,context,heads,ffn,dropout,layers)
        self.variant=variant
        if variant=='rope':
            self.position=None
            self.register_buffer('inv_freq',1/(10000**(torch.arange(0,embed//heads,2).float()/(embed//heads))),persistent=False)
        if variant=='rmsnorm':
            self.final_norm=RMSNorm(embed)
            for block in self.blocks: block.ln1=RMSNorm(embed); block.ln2=RMSNorm(embed)
        if variant=='swiglu':
            for block in self.blocks: block.ffn=SwiGLU(embed,max(1,2*ffn//3),layers)
        if variant=='gqa':
            for block in self.blocks:
                block.key=nn.Linear(embed,kv_heads*block.head_dim,bias=False)
                block.value=nn.Linear(embed,kv_heads*block.head_dim,bias=False)
                nn.init.normal_(block.key.weight,std=.02); nn.init.normal_(block.value.weight,std=.02)

    def hidden(self,tokens,positions):
        if self.variant=='rope': return self.embedding(tokens)
        return super().hidden(tokens,positions)

    def rotate(self,x,positions):
        angles=positions.float().unsqueeze(-1)*self.inv_freq
        cos,sin=angles.cos()[:,None].to(x.dtype),angles.sin()[:,None].to(x.dtype)
        even,odd=x[...,0::2],x[...,1::2]
        return torch.stack((even*cos-odd*sin,even*sin+odd*cos),dim=-1).flatten(-2)

    def qkv(self,block,h,positions):
        q,k,v=super().qkv(block,h,positions)
        if self.variant=='rope': q,k=self.rotate(q,positions),self.rotate(k,positions)
        return q,k,v

    def forward(self,tokens,segments=None,positions=None):
        if tokens.ndim!=2 or not 0<tokens.shape[1]<=self.block_size:
            raise ValueError('tokens must have shape (B,T) within block_size')
        if segments is None and positions is None:
            segments=torch.zeros_like(tokens)
            positions=torch.arange(tokens.shape[1],device=tokens.device)[None,:].expand_as(tokens)
        if segments is None or positions is None or segments.shape!=tokens.shape or positions.shape!=tokens.shape:
            raise ValueError('packed segments/positions must match tokens')
        valid=tokens.ne(0); mask=valid.unsqueeze(-1); length=tokens.shape[1]
        causal=torch.ones(length,length,dtype=torch.bool,device=tokens.device).tril()
        allowed=(causal[None,None]&valid[:,None,:,None]&valid[:,None,None,:]
                 &segments[:,None,:,None].eq(segments[:,None,None,:]))
        h=self.hidden(tokens,positions)*mask
        for block in self.blocks:
            q,k,v=self.qkv(block,block.ln1(h),positions)
            groups=q.shape[1]//k.shape[1]
            k=k.repeat_interleave(groups,dim=1); v=v.repeat_interleave(groups,dim=1)
            scores=(q@k.transpose(-2,-1))/math.sqrt(block.head_dim)
            scores=scores.masked_fill(~allowed,float('-inf')).masked_fill(~allowed.any(-1,keepdim=True),0)
            weights=scores.softmax(-1).masked_fill(~allowed,0)
            merged=(weights@v).transpose(1,2).reshape(tokens.shape[0],length,-1)
            h=(h+block.dropout(block.projection(merged)*mask))*mask
            h=(h+block.dropout(block.ffn(block.ln2(h))))*mask
        logits=self.head(self.final_norm(h)*mask)
        ids=torch.arange(logits.shape[-1],device=tokens.device)
        return logits.masked_fill((ids<6)&ids.ne(1),float('-inf'))


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION='v0.4.5'
    VARIANT='baseline'
    KV_HEADS=2
    EMBED,LAYERS,FFN_HIDDEN,EPOCHS=64,4,256,4

    def build_net(self):
        net=ModernNetwork(len(self.itos),self.EMBED,self.BLOCK_SIZE,self.HEADS,self.FFN_HIDDEN,
                          self.DROPOUT,self.LAYERS,self.VARIANT,self.KV_HEADS)
        if self.TIE_WEIGHTS: net.head.weight=net.embedding.weight
        return net

    def training_config(self):
        return {**super().training_config(),'VARIANT':self.VARIANT,'KV_HEADS':self.KV_HEADS}

    def extra_metadata(self):
        return {**super().extra_metadata(),'variant':self.VARIANT,'kv_heads':self.KV_HEADS}

    def restore_extra_metadata(self,meta):
        super().restore_extra_metadata(meta)
        self.VARIANT,self.KV_HEADS=meta['variant'],meta['kv_heads']


Model=NGramLM=NeuralLM
MODEL_PATH,VOCAB_PATH=str(HERE/'model.pt'),str(HERE/'vocab.json')
DATA_PATH,VALID_PATH=previous.DATA_PATH,previous.VALID_PATH
