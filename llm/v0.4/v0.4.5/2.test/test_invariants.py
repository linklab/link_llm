"""All architecture variants must preserve causal, packed, cached and save contracts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v045_tests',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def model(variant):
    lm=m.Model(); lm.DEVICE='cpu'; lm.VARIANT=variant; lm.EMBED=16; lm.HEADS=4
    lm.KV_HEADS=2; lm.LAYERS=2; lm.FFN_HIDDEN=32; lm.BLOCK_SIZE=8
    lm.VOCAB_SIZE=270; lm.DROPOUT=0; torch.manual_seed(17)
    lm.initialize(['abc 한글']); lm.net.eval(); return lm


class ModernTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_every_variant_packed_causal_and_gradients(self):
        for variant in m.VARIANTS:
            with self.subTest(variant=variant):
                lm=model(variant)
                x=torch.tensor([[2,103,104,2,105,106,0,0]])
                segments=torch.tensor([[0,0,0,1,1,1,-1,-1]])
                positions=torch.tensor([[0,1,2,0,1,2,0,0]])
                logits=lm.net(x,segments,positions)
                for start in (0,3):
                    torch.testing.assert_close(logits[:,start:start+3],lm.net(x[:,start:start+3]),rtol=1e-5,atol=1e-6)
                changed=x.clone(); changed[0,1]=110; changed[0,5]=111
                other=lm.net(changed,segments,positions)
                torch.testing.assert_close(logits[:,3:5],other[:,3:5],rtol=0,atol=0)
                y=torch.tensor([[103,104,1,105,106,1,-100,-100]])
                loss=m.sequence_loss(logits,y); loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in lm.net.parameters() if p.grad is not None))
                self.assertTrue(torch.isneginf(logits[...,0]).all())

    def test_every_variant_cache_and_checkpoint(self):
        for variant in m.VARIANTS:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                lm=model(variant); cache=None
                tokens=[2]+list(range(103,113))
                for end in range(1,len(tokens)+1):
                    x=torch.tensor([tokens[max(0,end-8):end]])
                    logits,cache=lm.net.cached(x,cache)
                    torch.testing.assert_close(logits,lm.net(x)[:,-1],rtol=1e-5,atol=1e-6)
                self.assertEqual(cache['layers'][0][0].shape[1],2 if variant=='gqa' else 4)
                lm.save(Path(d)/'model.pt'); restored=model('baseline'); restored.load(Path(d)/'model.pt')
                self.assertEqual(restored.VARIANT,variant)
                self.assertEqual(lm.evaluate(['test']),restored.evaluate(['test']))

    def test_rope_rotation_norm_and_relative_positions(self):
        net=model('rope').net; x=torch.randn(1,4,3,4); y=torch.randn_like(x)
        pos=torch.tensor([[0,1,2]])
        rot=net.rotate(x,pos)
        torch.testing.assert_close(rot.square().sum(-1),x.square().sum(-1))
        first=rot@net.rotate(y,pos).transpose(-1,-2)
        shifted=net.rotate(x,pos+7)@net.rotate(y,pos+7).transpose(-1,-2)
        torch.testing.assert_close(first,shifted,rtol=1e-5,atol=1e-6)

    def test_rmsnorm_and_swiglu_reference(self):
        x=torch.randn(2,3,16); norm=m.RMSNorm(16)
        torch.testing.assert_close(norm(x),x/(x.square().mean(-1,keepdim=True)+1e-5).sqrt())
        self.assertTrue(torch.isfinite(norm(torch.zeros_like(x))).all())
        f=m.SwiGLU(16,21,2)
        torch.testing.assert_close(f(x),f.down(torch.nn.functional.silu(f.gate(x))*f.up(x)))

    def test_baseline_unchanged_and_invalid_variant(self):
        lm=model('baseline'); old=m.previous.Model(); old.DEVICE='cpu'
        for key in ('EMBED','HEADS','LAYERS','FFN_HIDDEN','BLOCK_SIZE','VOCAB_SIZE','DROPOUT'):
            setattr(old,key,getattr(lm,key))
        old.initialize(['abc 한글']); old.net.load_state_dict(lm.net.state_dict()); old.net.eval()
        x=torch.tensor([[2,103,104]])
        torch.testing.assert_close(lm.net(x),old.net(x),rtol=0,atol=0)
        with self.assertRaises(ValueError): m.ModernNetwork(270,16,8,4,32,0,2,'gqa',3)
        with self.assertRaises(ValueError): m.ModernNetwork(270,12,8,4,32,0,2,'rope')

    def test_each_variant_exact_resume(self):
        for variant in m.VARIANTS:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                full=resume_tests.model(); full.VARIANT=variant; full.KV_HEADS=1
                full.train(resume_tests.DOCS,resume_tests.VALID)
                partial=resume_tests.model(); partial.VARIANT=variant; partial.KV_HEADS=1
                p=Path(d)/'state.pt'
                partial.train(resume_tests.DOCS,resume_tests.VALID,checkpoint=p,max_steps=2)
                restored=resume_tests.model(); restored.VARIANT=variant; restored.KV_HEADS=1
                restored.train(resume_tests.DOCS,resume_tests.VALID,resume=p)
                self.assertEqual(full.step_history,restored.step_history)
                for key,value in full.net.state_dict().items():
                    torch.testing.assert_close(value,restored.net.state_dict()[key],rtol=0,atol=0)


resume_tests=m.module('v045_resume_tests',VERSION.parent/'v0.4.2/2.test/test_invariants.py')
resume_tests.m=m


class ModernResumeTests(resume_tests.ResumeTests):
    pass


if __name__=='__main__': unittest.main()
