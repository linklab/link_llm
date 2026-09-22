"""공통 비교 계약, 모델 경계, 저장·복원을 검증하는 CPU 테스트."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import torch

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).resolve().parents[1]/'0.model/comparison.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.train=['a b c','b a d','a c']
        self.valid=['a b d','a unknown c','b a d']
        self.reader=m.gpt.Model();self.reader.itos=self.reader.build_vocab(self.train);self.reader.stoi={t:i for i,t in enumerate(self.reader.itos)}

    def model(self,cls):
        lm=cls();lm.DEVICE='cpu';lm.BLOCK_SIZE=2;lm.EMBED=8;lm.HEADS=2;lm.LAYERS=2;lm.FFN_HIDDEN=16;lm.HIDDEN=16
        lm.initialize(self.train);lm.net.eval()
        return lm

    def test_neural_batched_scores_equal_actual_probabilities(self):
        for cls in (m.MLP,m.gpt.Model):
            lm=self.model(cls)
            expected=[math.log(lm.token_prob(c,t)) for c,t in m.positions(lm,self.valid,2)]
            lm.net.train()
            rows=m.scored(lm,self.valid)
            self.assertTrue(lm.net.training)
            for a,b in zip(expected,rows):self.assertAlmostEqual(a,b[0],places=5)
            self.assertAlmostEqual(m.metrics(rows)['ppl'],lm.perplexity(self.valid),places=4)
            self.assertEqual(len(rows),len(expected))
            self.assertLess(m.metrics(rows)['support'],1)

    def test_mlp_causal_and_left_padding(self):
        lm=self.model(m.MLP);net=lm.net
        x=torch.tensor([[1,2]])
        actual=net(x)
        for i in range(2):
            context=torch.nn.functional.pad(x[:,:i+1],(1-i,0))
            expected=net.core(context)
            torch.testing.assert_close(actual[:,i,1:],expected[:,1:])
        y=x.clone();y[:,1]=3
        torch.testing.assert_close(actual[:,0],net(y)[:,0],rtol=0,atol=0)

    def test_checkpoint_roundtrip_nondefault(self):
        for cls in (m.MLP,m.gpt.Model):
            lm=self.model(cls)
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'model.pt';lm.save(p)
                restored=cls();restored.DEVICE='cpu';restored.load(p)
                self.assertEqual(restored.BLOCK_SIZE,2)
                torch.testing.assert_close(lm.net(torch.tensor([[1,2]])),restored.net(torch.tensor([[1,2]])),rtol=0,atol=0)

    def test_normalized_count_without_target_backoff(self):
        lm=m.Count();lm.ORDERS=[1,2];lm.BLOCK_SIZE=2
        lm.known=set('abcd');lm.stoi={t:i for i,t in enumerate('abcd')}
        lm.model=lm.to_dict({'2':{'a b':{'c':2}},'1':{'b':{'c':2,'d':1}}})
        self.assertEqual(sum(lm.next_dist(['a','b']).values()),1)
        self.assertEqual(lm.token_prob(['a','b'],'d'),m.FLOOR)
        self.assertEqual(lm.next_dist(['unknown','b']),None)
        self.assertAlmostEqual(lm.token_prob(['b'],'d'),1/3)

    def test_novelty_partitions_and_empty_bucket(self):
        labels=m.novelty(self.reader,['a b c'],['a b c','a b d','d a'],2)
        self.assertTrue(all(b in labels for b in m.BUCKETS))
        rows=[(-1,True,True,True)]*len(labels)
        result=m.summarize(rows,labels)
        self.assertEqual(sum(v['n'] for v in result['buckets'].values()),result['all']['n'])
        self.assertIsNone(m.summarize([],[])['all']['ppl'])
        json.dumps(m.summarize([],[]),allow_nan=False)
        with self.assertRaises(ValueError):m.summarize(rows,labels[:-1])

    def test_alignment_checks_tokens_not_just_count(self):
        class Wrong:
            def tokenize(self,text):return text.split()[::-1]
            def prepare(self,tokens):return tokens+['<END>']
        with self.assertRaises(ValueError):m.assert_alignment(self.reader,Wrong(),['a b c'])

    def test_track_selection_does_not_mix_seeds(self):
        rows=[]
        for seed in (1,2):
            for kind,context in m.candidates(3):
                rows.append({'id':f'{kind}-{context}-{seed}','kind':kind,'seed':seed,'context':context,
                             'scores':{'all':{'ppl':10/context if seed==1 else context}}})
        selected=m.select(rows,3)
        self.assertEqual(len(selected['matched_context']),6)
        self.assertIn('gpt-32-1',selected['best_in_declared_search'])
        self.assertIn('gpt-3-2',selected['best_in_declared_search'])
        with self.assertRaises(ValueError):m.select([r for r in rows if r['kind']!='mlp'],3)

    def test_seeded_generation_and_empty_detection(self):
        lm=self.model(m.MLP)
        a=m.generate(lm,'a',0.7,42,4);b=m.generate(lm,'a',0.7,42,4)
        self.assertEqual(a,b)
        self.assertTrue(m.generate(lm,'unknown',0,42)['empty'])
        self.assertNotIn(lm.PAD,a['tokens'])

    def test_full_cli_small_data_and_replay(self):
        version=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=root/'train.txt';valid=root/'valid.txt';out=root/'output'
            train.write_text('a b c\nb a d\na c\n')
            valid.write_text('a b d\nb a c\n')
            subprocess.run([sys.executable,str(version/'1.train/train.py'),
                '--train',str(train),'--valid',str(valid),'--output-dir',str(out),
                '--epochs','1','--embed','8','--hidden','16','--layers','1','--heads','2'],
                check=True,capture_output=True,text=True)
            report=json.loads((out/'comparison_report.json').read_text())
            self.assertEqual(len(report['runs']),6)
            for r in report['runs']:
                self.assertGreater(r['scores']['all']['support'],0)
                self.assertLess(r['scores']['all']['ppl'],1/m.FLOOR)
            subprocess.run([sys.executable,str(version/'2.test/test.py'),
                '--report',str(out/'comparison_report.json'),'--train',str(train),
                '--valid',str(valid)],check=True,capture_output=True,text=True)
            self.assertTrue((out/'model.pt').exists())


if __name__=='__main__':unittest.main(verbosity=2)
