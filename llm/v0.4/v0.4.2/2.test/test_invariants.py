"""Actual interrupted training must equal uninterrupted training, including dropout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v042_tests', VERSION/'0.model/lm.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
DOCS = ['가나다 abc '+str(i)*12 for i in range(8)]
VALID = ['unseen 검증 문서']


def model():
    lm = m.Model()
    for k,v in dict(DEVICE='cpu', EMBED=8, HEADS=2, LAYERS=2, FFN_HIDDEN=16,
                    BLOCK_SIZE=8, VOCAB_SIZE=270, EPOCHS=2, BATCH_SIZE=3,
                    DROPOUT=.2, WARMUP_STEPS=2).items():
        setattr(lm,k,v)
    return lm


class ResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_exact_mid_epoch_resume_and_inference_reload(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt'
            full=model(); full.train(DOCS,VALID,checkpoint=Path(d)/'full.pt')
            partial=model(); partial.train(DOCS,VALID,checkpoint=path,max_steps=2)
            self.assertFalse(partial.completed)
            resumed=model(); resumed.train(DOCS,VALID,resume=path,checkpoint=path)
            self.assertTrue(resumed.completed)
            self.assertEqual(full.history,resumed.history)
            self.assertEqual(full.step_history,resumed.step_history)
            for k,v in full.net.state_dict().items():
                torch.testing.assert_close(v,resumed.net.state_dict()[k],rtol=0,atol=0)
            a=torch.load(Path(d)/'full.pt',weights_only=True)
            b=torch.load(path,weights_only=True)
            for key in ('torch_rng','sampler_rng'):
                self.assertTrue(torch.equal(a[key],b[key]))
            self.assertEqual(a['scheduler'],b['scheduler'])
            for i,st in a['optimizer']['state'].items():
                for k,v in st.items():
                    torch.testing.assert_close(v,b['optimizer']['state'][i][k],rtol=0,atol=0)
            resumed.save(Path(d)/'model.pt')
            loaded=model(); loaded.load(Path(d)/'model.pt')
            self.assertEqual(full.evaluate(VALID),loaded.evaluate(VALID))
            self.assertEqual(sum(s['tokens'] for s in full.step_history),
                             full.packing_stats['targets']*full.EPOCHS)
            self.assertLess(full.step_history[-1]['lr'],full.LR)

    def test_resume_rejects_changed_data_and_horizon(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.pt'; lm=model(); lm.train(DOCS,VALID,checkpoint=p,max_steps=1)
            changed=model(); changed.EPOCHS=3
            with self.assertRaisesRegex(ValueError,'mismatch'):
                changed.train(DOCS,VALID,resume=p)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                model().train(DOCS+['extra'],VALID,resume=p)

    def test_invalid_config_and_duplicate_guard(self):
        lm=model(); lm.GRAD_CLIP=0
        with self.assertRaises(ValueError): lm.train(DOCS,VALID)
        with self.assertRaises(ValueError): model().train(DOCS,[DOCS[0]])


if __name__ == '__main__':
    unittest.main()
