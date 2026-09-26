"""BPE 가역성, 학습 split 격리, 저장 복원 및 바이트 채점 계약."""
import copy
import io
from contextlib import redirect_stdout
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v040_test',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Invariants(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.train=['사과 바나나','사과 사과','  a b  ']
        self.lm=m.Model();self.lm.DEVICE='cpu';self.lm.VOCAB_SIZE=280
        self.lm.BLOCK_SIZE=4;self.lm.EMBED=8;self.lm.HEADS=2;self.lm.LAYERS=2;self.lm.FFN_HIDDEN=16
        self.lm.initialize(self.train);self.lm.net.eval()

    def test_roundtrip_unseen_unicode_spaces_and_literals(self):
        for text in ['', '  사과\t\n망고 🥭\r', '<EOS><PAD><USER>', 'e\u0301é', '漢字𐍈', '\x00', '  ']:
            ids=self.lm.bpe.encode(text)
            self.assertTrue(all(i>=6 for i in ids))
            self.assertEqual(self.lm.bpe.decode(ids),text)
            self.assertEqual(self.lm.detokenize(self.lm.tokenize(text)),text)
        self.assertEqual(self.lm.bpe.special_id('<EOS>'),1)
        self.assertEqual(self.lm.bpe.decode([1]),'<EOS>')
        self.assertEqual(self.lm.bpe.decode([1],skip_special=True),'')

    def test_bpe_merge_and_determinism(self):
        a=m.ByteBPE.train(['aaaa']*4,264)
        b=m.ByteBPE.train(['aaaa']*4,264)
        self.assertEqual(a.to_dict(),b.to_dict())
        self.assertLess(len(a.encode('aaaa')),4)
        self.assertEqual(a.decode(a.encode('aaaa')),'aaaa')
        for size in (0,261,1.5):
            with self.assertRaises(ValueError):m.ByteBPE.train(['a'],size)

    def test_training_split_only(self):
        lm=self.lm;lm.EPOCHS=1
        lm.train(self.train,['망고 🥭']*2)
        self.assertEqual(lm.bpe.to_dict(),m.ByteBPE.train(self.train,280).to_dict())
        self.assertEqual(lm.bpe.training_hash,m.tokenizer.corpus_hash(self.train))
        self.assertEqual(lm.evaluate(['망고 🥭'])['coverage'],1)

    def test_windows_score_first_token_and_eos_once(self):
        for text in ['','a','사과 바나나 망고 🥭']:
            rows=self.lm.make_windows([text]);actual=[]
            for x,y,_ in rows:
                actual.extend((x[:j+1],t) for j,t in enumerate(y) if t!=-100)
            ids=[self.lm.stoi[t] for t in self.lm.prepare(self.lm.tokenize(text))]
            expected=[(ids[max(0,i-4):i],ids[i]) for i in range(1,len(ids))]
            self.assertEqual(actual,expected)
            self.assertEqual(actual[0][0],[2])
            self.assertEqual(actual[-1][1],1)

    def test_exact_nll_byte_denominator_and_no_floor(self):
        docs=[' a  ', '망고 🥭', '']
        with torch.no_grad():
            self.lm.net.head.bias[self.lm.bpe.encode('a')[0]]=-50
        expected_text=expected_eos=0.;n=0
        for doc in docs:
            ids=[self.lm.stoi[t] for t in self.lm.prepare(self.lm.tokenize(doc))]
            for i in range(1,len(ids)):
                x=torch.tensor([ids[max(0,i-4):i]])
                nll=-self.lm.net(x)[0,-1].double().log_softmax(-1)[ids[i]].item()
                if ids[i]==1:expected_eos+=nll
                else:expected_text+=nll
                n+=1
        self.lm.net.train();r=self.lm.evaluate(docs)
        self.assertTrue(self.lm.net.training)
        self.assertAlmostEqual(r['text_nll'],expected_text,places=5)
        self.assertAlmostEqual(r['eos_nll'],expected_eos,places=5)
        self.assertEqual(r['n'],n)
        self.assertEqual(r['text_bytes'],sum(len(s.encode('utf-8')) for s in docs))
        self.assertAlmostEqual(r['nll_per_byte'],expected_text/r['text_bytes'],places=6)
        self.assertAlmostEqual(r['bits_per_byte'],r['nll_per_byte']/math.log(2))
        self.assertIsNone(self.lm.evaluate([''])['nll_per_byte'])
        self.assertIsNone(self.lm.evaluate([])['nll_per_byte'])
        self.assertGreater(r['ppl'],math.exp((expected_eos+expected_text-20)/n))

    def test_masks_causality_and_gradient(self):
        x=torch.tensor([[2,10,11,12]])
        y=x.clone();y[0,-1]=13
        torch.testing.assert_close(self.lm.net(x)[:,:3],self.lm.net(y)[:,:3],rtol=0,atol=0)
        logits=self.lm.net(x)
        self.assertTrue(torch.isneginf(logits[:,:,[0,2,3,4,5]]).all())
        self.assertTrue(torch.isfinite(logits[:,:,6:]).all())
        loss=m.sequence_loss(logits,torch.tensor([[10,11,12,1]]));loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.lm.net.parameters()))
        self.assertNotIn(self.lm.USER,self.lm.next_dist([self.lm.BOS]))

    def test_checkpoint_and_tokenizer_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt';self.lm.save(path)
            restored=m.Model();restored.DEVICE='cpu'
            with patch.object(torch.backends.mps,'is_available',return_value=True):restored.load(path)
            self.assertEqual(restored.tokenize('망고 🥭'),self.lm.tokenize('망고 🥭'))
            x=torch.tensor([[2,10,11]])
            torch.testing.assert_close(restored.net(x),self.lm.net(x),rtol=0,atol=0)
            self.assertEqual(m.ByteBPE.load(Path(tmp)/'tokenizer.json').to_dict(),self.lm.bpe.to_dict())
            self.assertTrue((Path(tmp)/'config.json').exists())
            # 모델 파일만 있어도 토크나이저를 복원합니다.
            (Path(tmp)/'tokenizer.json').unlink();(Path(tmp)/'vocab.json').unlink()
            restored.load(path)
            checkpoint=torch.load(path,weights_only=True)
            checkpoint['meta']['bpe']['specials'][1]='BROKEN';torch.save(checkpoint,path)
            with self.assertRaises(ValueError):m.Model().load(path)

    def test_corrupt_tokenizer_rejected(self):
        d=copy.deepcopy(self.lm.bpe.to_dict());d['merges'][0]=[0,999]
        with self.assertRaises(ValueError):m.ByteBPE.from_dict(d)
        with self.assertRaises(ValueError):self.lm.bpe.decode([-1])

    def test_generation_empty_prompt_eos_and_limit(self):
        with patch.object(self.lm,'next_token',return_value=self.lm.END):
            self.assertEqual(self.lm.generate('망고'),'망고')
            self.assertEqual(self.lm.generate(''),'')
        token=self.lm.tokenize('x')[0];self.lm.MAX_LENGTH=3
        with patch.object(self.lm,'next_token',return_value=token):self.assertEqual(self.lm.generate(''),'xxx')
        with self.assertRaises(NotImplementedError):self.lm.chat('안녕')

    def test_reader_preserves_raw_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.txt';p.write_bytes(' a \r\n\n한글\n'.encode('utf-8'))
            self.assertEqual(self.lm.read_sentences(p),[' a \r','','한글'])

    def test_legacy_suite_excludes_new_contract(self):
        suite=m.module('legacy_eval',m.ROOT/'eval_suite.py')
        with patch.object(suite,'load_model',return_value=self.lm):
            rows,skipped=suite.collect(['v0.4.0'],[],[],[],True)
        self.assertEqual(rows,[]);self.assertEqual(len(skipped),1)
        output=io.StringIO()
        with redirect_stdout(output):suite.report(rows,skipped,[],[],True)
        self.assertIn('별도 토크나이저/평가 계약',output.getvalue())

    def test_cli_train_save_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=root/'train.txt';valid=root/'valid.txt';out=root/'out'
            train.write_text('a b\na c\n');valid.write_text('a 망고\n')
            subprocess.run([sys.executable,str(VERSION/'1.train/train.py'),'--train',str(train),
                '--valid',str(valid),'--output-dir',str(out),'--epochs','1','--vocab-size','270',
                '--embed','8','--heads','2','--layers','1','--ffn-hidden','16','--device','cpu'],
                check=True,capture_output=True,text=True)
            restored=m.Model();restored.DEVICE='cpu';restored.load(out/'model.pt')
            import json
            report=json.loads((out/'training_report.json').read_text())
            result=restored.evaluate(restored.read_sentences(valid))
            self.assertAlmostEqual(result['nll_per_byte'],report['valid']['nll_per_byte'],places=6)
            self.assertEqual(report['scored_documents']['valid'],m.tokenizer.corpus_hash(['a 망고']))
            command=[sys.executable,str(VERSION/'2.test/test.py'),'--model',str(out/'model.pt'),
                     '--valid',str(valid),'--prompt','a']
            subprocess.run(command,check=True,capture_output=True,text=True)
            valid.write_text('다른 원문\n')
            failed=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('검증 원문',failed.stderr)
            subprocess.run(command+['--allow-new-data'],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main(verbosity=2)
