"""Cached/full logits, eviction and interleaved independent requests."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v044_tests',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def model():
    lm=m.Model(); lm.DEVICE='cpu'; lm.EMBED=16; lm.HEADS=4; lm.LAYERS=2
    lm.FFN_HIDDEN=32; lm.BLOCK_SIZE=8; lm.VOCAB_SIZE=270; lm.DROPOUT=.2
    torch.manual_seed(17); lm.initialize(['abc 한글']); lm.net.eval(); return lm


class CacheTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_append_eviction_and_changed_prefix_match_full(self):
        lm=model(); cache=None; tokens=[2,103,104,105,106,107,108,109,110,111,112]
        for end in range(1,len(tokens)+1):
            x=torch.tensor([tokens[max(0,end-8):end]])
            actual,cache=lm.net.cached(x,cache)
            torch.testing.assert_close(actual,lm.net(x)[:,-1],rtol=1e-5,atol=1e-6)
            self.assertEqual(cache['reused_prefix'],end-1 if end<=8 else 0)
        x[0,0]=115
        actual,cache=lm.net.cached(x,cache)
        self.assertEqual(cache['reused_prefix'],0)
        torch.testing.assert_close(actual,lm.net(x)[:,-1],rtol=1e-5,atol=1e-6)

    def test_interleaved_sessions_and_owner(self):
        lm=model(); a=b=None
        for length in (1,3,6,8):
            x=torch.tensor([[2]+[103]*(length-1)]); y=torch.tensor([[2]+[110]*(length-1)])
            left,a=lm.net.cached(x,a); right,b=lm.net.cached(y,b)
            torch.testing.assert_close(left,lm.net(x)[:,-1],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(right,lm.net(y)[:,-1],rtol=1e-5,atol=1e-6)
            self.assertIsNot(a,b)
        with self.assertRaisesRegex(ValueError,'another'): model().net.cached(x,a)

    def test_generate_greedy_seeded_and_raw_inputs(self):
        lm=model(); lm.MAX_LENGTH=15
        for prompt in ('',' ','한글','a'*40):
            for temperature in (0.,.8):
                self.assertEqual(lm.generate(prompt,temperature,top_k=7,top_p=.9,seed=44),
                                 lm.generate(prompt,temperature,top_k=7,top_p=.9,seed=44,use_cache=False))

    def test_sampling_transfers_logits_before_float64(self):
        # Enforce the MPS restriction even on CPU-only test runners.
        tensor_cpu,tensor_double=torch.Tensor.cpu,torch.Tensor.double
        transferred=set()
        conversions=[]

        def cpu(tensor,*args,**kwargs):
            result=tensor_cpu(tensor,*args,**kwargs)
            transferred.add(id(result))
            return result

        def double(tensor,*args,**kwargs):
            self.assertIn(id(tensor),transferred,'float64 attempted before CPU transfer')
            conversions.append(1)
            return tensor_double(tensor,*args,**kwargs)

        lm=model(); lm.MAX_LENGTH=3
        with patch.object(torch.Tensor,'cpu',cpu),patch.object(torch.Tensor,'double',double):
            for cached in (False,True):
                for temperature in (0.,.8):
                    lm.generate('아침에',temperature,use_cache=cached,seed=44)
        self.assertTrue(conversions)

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS hardware unavailable')
    def test_mps_generation_all_inheriting_versions(self):
        for version in ('v0.4.4','v0.4.5','v0.4.6'):
            implementation=m.module('mps_'+version,VERSION.parent/version/'0.model/lm.py')
            lm=implementation.Model(); lm.DEVICE='mps'; lm.EMBED=16; lm.HEADS=4
            lm.LAYERS=2; lm.FFN_HIDDEN=32; lm.BLOCK_SIZE=8; lm.VOCAB_SIZE=270
            lm.DROPOUT=0; lm.MAX_LENGTH=3
            if version!='v0.4.4': lm.VARIANT='rope'
            lm.initialize(['아침에 abc']); lm.net.eval()
            self.assertEqual(lm.device().type,'mps')
            for prompt in ('아침에','아','',' '):
                for cached in (False,True):
                    for temperature in (0.,.8):
                        with self.subTest(version=version,prompt=prompt,cached=cached,temperature=temperature):
                            result=lm.generate(prompt,temperature,use_cache=cached,seed=44)
                            self.assertTrue(result.startswith(prompt))

    def test_reject_training_padding_and_bad_shapes(self):
        lm=model()
        for x in (torch.tensor([[0]]),torch.ones(2,3,dtype=torch.long),torch.ones(1,9,dtype=torch.long)):
            with self.assertRaises(ValueError): lm.net.cached(x)
        lm.net.train()
        with self.assertRaises(ValueError): lm.net.cached(torch.tensor([[2]]))


resume_tests=m.module('v044_resume_tests', VERSION.parent/'v0.4.2/2.test/test_invariants.py')
resume_tests.m=m


class CacheResumeTests(resume_tests.ResumeTests):
    pass


if __name__=='__main__': unittest.main()
