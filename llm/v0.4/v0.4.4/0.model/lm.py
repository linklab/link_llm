"""Request-local KV caches; rebuild on sliding-window eviction for exact semantics."""
import importlib.util
import math
import random
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('v044_previous',HERE.parents[1]/'v0.4.3/0.model/lm.py')
previous=importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
module,corpus,packing=previous.module,previous.corpus,previous.packing
ByteBPE,sequence_loss=previous.ByteBPE,previous.sequence_loss


class CachedNetwork(previous.Network):
    def hidden(self,tokens,positions):
        return self.embedding(tokens)+self.position(positions)

    def qkv(self,block,h,positions):
        def split(layer):
            return layer(h).reshape(h.shape[0],h.shape[1],-1,block.head_dim).transpose(1,2)
        return split(block.query),split(block.key),split(block.value)

    @torch.no_grad()
    def cached(self,tokens,cache=None):
        """tokens is the entire current window. Returns last logits and a new cache.

        Append-only prefixes reuse K/V. Eviction or edited prefixes rebuild every
        layer, preserving learned absolute positions and full-window evaluation.
        Cache ownership is local to one network and one caller, never global.
        """
        if self.training:
            raise ValueError('KV cache requires eval mode')
        if tokens.ndim!=2 or tokens.shape[0]!=1 or not 0<tokens.shape[1]<=self.block_size or (tokens==0).any():
            raise ValueError('cache expects one non-PAD window within block_size')
        if cache is not None and cache['owner'] is not self:
            raise ValueError('cache belongs to another network')
        length=tokens.shape[1]
        reuse=(cache is not None and cache['tokens'].shape[1]<=length
               and torch.equal(cache['tokens'],tokens[:,:cache['tokens'].shape[1]]))
        offset=cache['tokens'].shape[1] if reuse else 0
        if offset==length:
            return cache['logits'],cache
        suffix=tokens[:,offset:]
        positions=torch.arange(offset,length,device=tokens.device)[None,:]
        h=self.hidden(suffix,positions)
        layers=[]
        for i,block in enumerate(self.blocks):
            q,k,v=self.qkv(block,block.ln1(h),positions)
            if reuse:
                oldk,oldv=cache['layers'][i]
                k=torch.cat((oldk,k),dim=2); v=torch.cat((oldv,v),dim=2)
            layers.append((k,v))
            groups=q.shape[1]//k.shape[1]
            keys=k.repeat_interleave(groups,dim=1)
            values=v.repeat_interleave(groups,dim=1)
            allowed=torch.arange(length,device=tokens.device)[None,:]<=positions[0,:,None]
            scores=(q@keys.transpose(-2,-1))/math.sqrt(block.head_dim)
            weights=scores.masked_fill(~allowed,float('-inf')).softmax(-1)
            merged=(weights@values).transpose(1,2).reshape(1,length-offset,-1)
            h=h+block.dropout(block.projection(merged))
            h=h+block.dropout(block.ffn(block.ln2(h)))
        logits=self.head(self.final_norm(h[:,-1]))
        ids=torch.arange(logits.shape[-1],device=tokens.device)
        logits=logits.masked_fill((ids<6)&ids.ne(1),float('-inf'))
        return logits,{'owner':self,'tokens':tokens.clone(),'layers':layers,'logits':logits,
                       'reused_prefix':offset}


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION='v0.4.4'

    def build_net(self):
        net=CachedNetwork(len(self.itos),self.EMBED,self.BLOCK_SIZE,self.HEADS,
                          self.FFN_HIDDEN,self.DROPOUT,self.LAYERS)
        if self.TIE_WEIGHTS: net.head.weight=net.embedding.weight
        return net

    def generate(self,start_text,temperature=0.,top_k=0,top_p=1.,use_cache=True,seed=None):
        if not math.isfinite(temperature) or temperature<0 or top_k<0 or not 0<top_p<=1:
            raise ValueError('invalid sampling parameters')
        ids=[2]+self.bpe.encode(start_text)
        cache=None
        rng=random.Random(seed)  # Sampling settings and RNG belong to this request.
        self.net.eval()
        for _ in range(self.MAX_LENGTH):
            window=torch.tensor([ids[-self.BLOCK_SIZE:]],device=self.device())
            with torch.no_grad():
                if use_cache: logits,cache=self.net.cached(window,cache)
                else: logits=self.net(window)[:,-1]
            logits=logits[0].double().cpu()
            if temperature<=.01:
                token=int(logits.argmax())
            else:
                probs=(logits/temperature).softmax(-1)
                order=probs.argsort(descending=True).tolist()
                if top_k: order=order[:top_k]
                if top_p<1:
                    kept=[]; cumulative=0.
                    for i in order:
                        kept.append(i); cumulative+=float(probs[i])
                        if cumulative>=top_p: break
                    order=kept
                token=rng.choices(order,weights=[float(probs[i]) for i in order],k=1)[0]
            if token==1: break
            ids.append(token)
        return self.bpe.decode(ids,skip_special=True,errors='replace')


Model=NGramLM=NeuralLM
MODEL_PATH,VOCAB_PATH=str(HERE/'model.pt'),str(HERE/'vocab.json')
DATA_PATH,VALID_PATH=previous.DATA_PATH,previous.VALID_PATH
