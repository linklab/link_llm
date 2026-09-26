"""Weight tying, extended context and checkpoint policy invariants."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v043_tests',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def model(tied=True):
    lm=m.Model(); lm.DEVICE='cpu'; lm.EMBED=16; lm.HEADS=4; lm.LAYERS=2
    lm.FFN_HIDDEN=32; lm.BLOCK_SIZE=64; lm.VOCAB_SIZE=270; lm.DROPOUT=0
    lm.TIE_WEIGHTS=tied; torch.manual_seed(17); lm.initialize(['abc 한글'])
    lm.net.eval(); return lm


class ScalingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_tied_identity_gradient_and_parameter_saving(self):
        tied,untied=model(),model(False)
        self.assertIs(tied.net.embedding.weight,tied.net.head.weight)
        self.assertEqual(sum(p.numel() for p in untied.net.parameters())-
                         sum(p.numel() for p in tied.net.parameters()),len(tied.itos)*tied.EMBED)
        loss=m.sequence_loss(tied.net(torch.tensor([[2,103,104]])),torch.tensor([[103,104,1]]))
        loss.backward()
        self.assertIs(tied.net.embedding.weight.grad,tied.net.head.weight.grad)
        self.assertGreater(tied.net.embedding.weight.grad.abs().sum().item(),0)

    def test_extended_context_influence_and_boundaries(self):
        lm=model(); x=torch.full((1,64),103); x[0,0]=2
        logits=lm.net(x); changed=x.clone(); changed[0,1]=104
        self.assertGreater((logits[0,-1,6:]-lm.net(changed)[0,-1,6:]).abs().max().item(),0)
        lm.net.zero_grad(); logits[0,-1,110].backward()
        self.assertGreater(lm.net.position.weight.grad[1].abs().sum().item(),0)
        with self.assertRaises(ValueError): lm.net(torch.full((1,65),103))
        self.assertEqual(len(lm._context_ids([lm.BOS]+lm.tokenize('a'*100))),64)

    def test_packed_isolation_and_reload(self):
        lm=model(); batch=m.packing.collate(m.packing.make_packs(lm,['abc','한글']))
        logits=lm.net(batch['x'],batch['segments'],batch['positions'])
        for i in (0,1):
            mask=batch['segments'][0]==i
            torch.testing.assert_close(logits[0,mask],lm.net(batch['x'][:,mask])[0],rtol=1e-5,atol=1e-6)
        with tempfile.TemporaryDirectory() as d:
            lm.save(Path(d)/'model.pt'); restored=model(False); restored.load(Path(d)/'model.pt')
            self.assertIs(restored.net.head.weight,restored.net.embedding.weight)
            self.assertEqual(lm.evaluate(['test']),restored.evaluate(['test']))


resume_tests=m.module('v043_resume_tests', VERSION.parent/'v0.4.2/2.test/test_invariants.py')
resume_tests.m=m


class ScalingResumeTests(resume_tests.ResumeTests):
    pass


if __name__=='__main__': unittest.main()
