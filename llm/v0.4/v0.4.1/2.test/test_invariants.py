"""Corpus isolation, packing semantics, inference parity and full CLI regression."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

VERSION = Path(__file__).resolve().parents[1]
ROOT = VERSION.parents[2]
spec = importlib.util.spec_from_file_location('v041_tests', VERSION/'0.model/lm.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
corpus = m.corpus
generator = m.module('v041_test_generator', ROOT/'data/pretrain/v0.4.1/generate.py')
SOURCES = {'s': {'uri':'repo:test-fixture', 'license':'MIT', 'usage':'unit test', 'description':'test'}}


def records():
    return [{'id': f'{i}-{j}', 'text': f'주제 {i} 기록 {j}\n\n끝. ', 'source':'s',
             'topic':str(i), 'template_id':f't{i}'} for i in range(9) for j in range(2)]


class Invariants(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.lm = m.Model(); self.lm.DEVICE='cpu'; self.lm.VOCAB_SIZE=270
        self.lm.EMBED=8; self.lm.HEADS=2; self.lm.LAYERS=2
        self.lm.FFN_HIDDEN=16; self.lm.BLOCK_SIZE=8; self.lm.DROPOUT=0
        self.lm.initialize(['a b c', '한글']); self.lm.net.eval()

    def test_every_target_once_and_chunk_context(self):
        docs = ['', 'a', ' a\n한글 🥭 '*4, 'z']
        packs = m.packing.make_packs(self.lm, docs)
        actual = [target for p in packs for target in p['y']]
        expected = [i for t in docs for i in self.lm.bpe.encode(t)+[1]]
        self.assertEqual(actual, expected)
        self.assertEqual(actual.count(1), len(docs))
        self.assertEqual(sum(p['x'].count(2) for p in packs), len(docs))
        chunks=[]
        for pack in packs:
            self.assertLessEqual(len(pack['x']), self.lm.BLOCK_SIZE)
            for seg in sorted(set(pack['segments'])):
                indexes=[i for i,s in enumerate(pack['segments']) if s==seg]
                self.assertEqual([pack['positions'][i] for i in indexes], list(range(len(indexes))))
                chunks.append(([pack['x'][i] for i in indexes], [pack['y'][i] for i in indexes]))
        reference=[]
        for text in docs:
            ids=[2]+self.lm.bpe.encode(text)+[1]
            for start in range(0,len(ids)-1,8):
                reference.append((ids[:-1][start:start+8],ids[1:][start:start+8]))
        self.assertEqual(chunks,reference)

    def test_packed_logits_equal_independent_chunks(self):
        packs=m.packing.make_packs(self.lm,['a','b','','c','long text abc '*3])
        batch=m.packing.collate(packs)
        packed=self.lm.net(batch['x'],batch['segments'],batch['positions'])
        for row, pack in enumerate(packs):
            for seg in sorted(set(pack['segments'])):
                ix=[i for i,s in enumerate(pack['segments']) if s==seg]
                separate=self.lm.net(torch.tensor([[pack['x'][i] for i in ix]]))
                torch.testing.assert_close(packed[row,ix],separate[0],rtol=1e-5,atol=1e-6)

    def test_document_isolation_causality_and_gradients(self):
        x=torch.tensor([[2,103,104,2,105,106,0,0]])
        seg=torch.tensor([[0,0,0,1,1,1,-1,-1]])
        pos=torch.tensor([[0,1,2,0,1,2,0,0]])
        first=self.lm.net(x,seg,pos)
        changed=x.clone(); changed[0,1:3]=torch.tensor([107,108])
        torch.testing.assert_close(first[:,3:6],self.lm.net(changed,seg,pos)[:,3:6],rtol=0,atol=0)
        changed=x.clone();changed[0,5]=109
        torch.testing.assert_close(first[:,:5],self.lm.net(changed,seg,pos)[:,:5],rtol=0,atol=0)
        y=torch.tensor([[-100,-100,-100,105,106,1,-100,-100]])
        m.sequence_loss(first,y).backward()
        self.assertEqual(self.lm.net.embedding.weight.grad[103:105].abs().sum().item(),0)
        self.assertGreater(self.lm.net.blocks[0].value.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.lm.net.parameters()))

    def test_padding_and_masks(self):
        x=torch.zeros((2,4),dtype=torch.long)
        logits=self.lm.net(x,torch.full_like(x,-1),x)
        self.assertTrue(torch.isfinite(logits[:,:,6:]).all())
        self.assertTrue(torch.isneginf(logits[:,:,[0,2,3,4,5]]).all())
        with self.assertRaises(ValueError):self.lm.net(x,torch.zeros((1,2)),x)

    def test_training_count_and_tokenizer_train_only(self):
        self.lm.EPOCHS=2
        train=['abc def','한글\n  보존','', 'a'*30]
        valid=['새 주제 🥭']
        self.lm.train(train,valid)
        self.assertEqual(self.lm.bpe.to_dict(),m.ByteBPE.train(train,270).to_dict())
        self.assertEqual(self.lm.packing_stats['targets'],sum(len(self.lm.bpe.encode(t))+1 for t in train))
        self.assertEqual(self.lm.packing_stats['targets'],self.lm.packing_stats['packed_input_tokens'])
        self.assertGreater(self.lm.packing_stats['sliding_input_tokens'],self.lm.packing_stats['packed_input_tokens'])

    def test_direct_api_rejects_duplicate_validation(self):
        for train,valid in [(['a'],['a\n']),(['가'],['가']),(['a b'],['a  b'])]:
            with self.assertRaisesRegex(ValueError,'duplicate'):self.lm.train(train,valid)

    def test_checkpoint_inference_and_evaluation_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            self.lm.corpus_manifest_sha256='fixture'
            self.lm.save(path)
            restored=m.Model();restored.DEVICE='cpu';restored.load(path)
            self.assertEqual(restored.corpus_manifest_sha256,'fixture')
            torch.testing.assert_close(restored.net(torch.tensor([[2,100]])),self.lm.net(torch.tensor([[2,100]])),rtol=0,atol=0)
            old=m.previous.Model();old.DEVICE='cpu'
            for key in ['EMBED','HEADS','LAYERS','FFN_HIDDEN','BLOCK_SIZE','DROPOUT']:
                setattr(old,key,getattr(self.lm,key))
            old.bpe=self.lm.bpe;old.initialize(['a']);old.net.load_state_dict(self.lm.net.state_dict());old.net.eval()
            self.assertEqual(restored.evaluate([' a\n🥭','']),old.evaluate([' a\n🥭','']))
            checkpoint=torch.load(path,weights_only=True)
            checkpoint['meta']['packing_policy']='broken';torch.save(checkpoint,path)
            with self.assertRaises(ValueError):restored.load(path)

    def test_corpus_reproducibility_provenance_and_group_disjointness(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            manifest=corpus.prepare(records(),SOURCES,a)
            corpus.prepare(list(reversed(records())),SOURCES,b)
            for name in ['manifest.json','train.jsonl','valid.jsonl','test.jsonl']:
                self.assertEqual((a/name).read_bytes(),(b/name).read_bytes())
            _,splits=corpus.load_bundle(a)
            for key in ['topic','template_id','group_id']:
                sets=[{r[key] for r in splits[n]} for n in corpus.SPLITS]
                self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            self.assertEqual(manifest['deduplication']['input_documents'],18)
            self.assertEqual({r['text'] for rows in splits.values() for r in rows},{r['text'] for r in records()})

    def test_dedup_links_groups_before_removal(self):
        rows=records()
        rows[0]['text']='  é\n';rows[2]['text']='e\u0301'
        with tempfile.TemporaryDirectory() as tmp:
            manifest=corpus.prepare(rows,SOURCES,tmp)
            _,splits=corpus.load_bundle(tmp)
            self.assertEqual(manifest['deduplication']['kept_documents'],17)
            owners={r['topic']:name for name,rs in splits.items() for r in rs}
            self.assertEqual(owners['0'],owners['1'])
            self.assertIn('  é\n',[r['text'] for rs in splits.values() for r in rs])

    def test_reject_unpartitionable_groups_and_bad_sources(self):
        rows=records()
        for r in rows:r['template_id']='shared'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'3 independent'):corpus.prepare(rows,SOURCES,tmp)
            with self.assertRaises(ValueError):corpus.prepare(records(),{'s':{}},tmp)

    def test_tampered_bundle_and_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);corpus.prepare(records(),SOURCES,root)
            (root/'valid.jsonl').write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'hash'):corpus.load_bundle(root)
            (root/'bad.jsonl').write_text('\n')
            with self.assertRaises(ValueError):corpus.read_jsonl(root/'bad.jsonl')

    def test_jsonl_preserves_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=records();rows[0]['text']='  가\u2028나\u0085각\r\n🥭  '
            corpus.prepare(rows,SOURCES,tmp)
            _,splits=corpus.load_bundle(tmp)
            recovered={r['id']:r['text'] for rs in splits.values() for r in rs}
            self.assertEqual(recovered[rows[0]['id']],rows[0]['text'])

    def test_declared_template_leakage_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=corpus.prepare(records(),SOURCES,root)
            train=corpus.read_jsonl(root/'train.jsonl');valid=corpus.read_jsonl(root/'valid.jsonl')
            valid[0]['template_id']=train[0]['template_id']
            raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in valid).encode()
            (root/'valid.jsonl').write_bytes(raw)
            manifest['splits']['valid']['sha256']=corpus.sha256(raw)
            (root/'manifest.json').write_bytes(corpus.json_bytes(manifest))
            with self.assertRaisesRegex(ValueError,'leakage'):corpus.load_bundle(root)

    def test_web_preserves_raw_and_empty_prompts_and_legacy_rules(self):
        server=m.module('v041_web_test',ROOT/'web_service/server.py')
        for prompt in ['  가\n ', '', '  ']:
            raw=json.dumps({'version':'v0.4.1','message':prompt}).encode()
            handler=object.__new__(server.Handler)
            handler.path='/api/chat';handler.headers={'Content-Length':str(len(raw))};handler.rfile=io.BytesIO(raw)
            responses=[];handler._send_json=lambda data,status=200: responses.append((status,data))
            with patch.object(server,'find_versions',return_value=['v0.4.1']),patch.object(server,'load_lm',return_value=self.lm),patch.object(server,'generate_reply',return_value='ok') as generate:
                handler.do_POST()
                self.assertEqual(generate.call_args.args[1],prompt)
                self.assertEqual(responses[0][0],200)
        raw=json.dumps({'version':'old','message':' '}).encode()
        handler.headers={'Content-Length':str(len(raw))};handler.rfile=io.BytesIO(raw)
        with patch.object(server,'find_versions',return_value=['old']),patch.object(server,'load_lm',return_value=object()):
            handler.do_POST()
        self.assertEqual(responses[-1][0],400)

    def test_v040_cli_rejects_final_lf_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'train').write_bytes(b'a b\na c\n');(p/'valid').write_bytes(b'a b\na c')
            r=subprocess.run([sys.executable,str(VERSION.parent/'v0.4.0/1.train/train.py'),
                              '--train',str(p/'train'),'--valid',str(p/'valid'),'--output-dir',str(p/'out')],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('원문이 같으면',r.stderr)
            self.assertFalse((p/'out').exists())

    def test_cli_training_reload_and_explicit_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';out=root/'model'
            corpus.prepare(records(),SOURCES,data)
            command=[sys.executable,str(VERSION/'1.train/train.py'),'--corpus-dir',str(data),
                     '--output-dir',str(out),'--epochs','1','--vocab-size','270','--embed','8',
                     '--heads','2','--layers','1','--ffn-hidden','16','--device','cpu','--threads','2']
            subprocess.run(command,check=True,capture_output=True,text=True)
            report=json.loads((out/'training_report.json').read_bytes())
            self.assertIn('not evaluated',report['test']['status'])
            self.assertEqual(report['tokenizer']['training_hash'],report['data']['train']['documents_hash'])
            lm=m.Model();lm.DEVICE='cpu';lm.load(out/'model.pt')
            actual=lm.evaluate(lm.read_sentences(data/'valid.jsonl'))
            self.assertAlmostEqual(actual['ppl'],report['valid']['ppl'],places=5)
            before=(out/'model.pt').read_bytes()
            evaluate=[sys.executable,str(VERSION/'2.test/test.py'),'--model',str(out/'model.pt'),
                      '--corpus-dir',str(data),'--split','test','--output-report',str(root/'test.json')]
            subprocess.run(evaluate,check=True,capture_output=True,text=True)
            subprocess.run(evaluate,check=True,capture_output=True,text=True)
            result=json.loads((root/'test.json').read_bytes())
            self.assertEqual(result['split'],'test');self.assertEqual(before,(out/'model.pt').read_bytes())
            (out/'model.pt').write_bytes(before+b'tampered')
            fail=subprocess.run(evaluate,capture_output=True,text=True)
            self.assertNotEqual(fail.returncode,0)
            self.assertIn('checkpoint does not match',fail.stderr)


if __name__=='__main__':
    unittest.main(verbosity=2)
