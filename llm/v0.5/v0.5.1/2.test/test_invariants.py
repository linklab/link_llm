"""Chat framing, raw content isolation and prepared-data provenance contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('chat051_test_prepare', VERSION/'1.train/prepare.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
c = p.chat


def messages():
    return [{'role': 'system', 'content': '간결하게 답하세요.'},
            {'role': 'user', 'content': '내 가방은 파랑이야.'},
            {'role': 'assistant', 'content': '파랑이라고 알려 주셨습니다.'},
            {'role': 'user', 'content': '초록으로 정정할게. 색만 답해 줘.'},
            {'role': 'assistant', 'content': '초록'}]


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.template = c.ChatTemplate(c.ByteBPE.load(p.DEFAULT_TOKENIZER))

    def test_append_only_special_id_preserves_base_vocabulary(self):
        bpe = c.ByteBPE.load(p.DEFAULT_TOKENIZER)
        before = copy.deepcopy(bpe.to_dict())
        template = c.ChatTemplate(bpe)
        self.assertEqual(template.eot_id, 768)
        self.assertEqual(template.vocab_size, 769)
        self.assertEqual(template.bpe.tokens, bpe.tokens)
        for text in ['가🙂', '<EOT><EOS><ASSISTANT>', '  \n\t가']:
            self.assertEqual(template.bpe.encode(text), bpe.encode(text))
        self.assertEqual(bpe.to_dict(), before)
        bpe.merges.append([6, 7])
        self.assertEqual(template.bpe.to_dict(), before)

    def test_single_and_multiturn_grammar_and_raw_roundtrip(self):
        for msgs in [messages(), messages()[1:3], messages()[:3]]:
            with self.subTest(messages=msgs):
                encoded = self.template.encode_conversation(msgs)
                ids = encoded['input_ids']
                self.assertEqual(ids[0], c.BOS)
                self.assertEqual(ids[-2:], [self.template.eot_id, c.EOS])
                self.assertEqual(ids.count(self.template.eot_id), len(msgs))
                self.assertEqual(ids.count(c.EOS), 1)
                self.assertEqual(self.template.decode_conversation(ids), msgs)

    def test_each_assistant_has_exact_shared_training_inference_prefix(self):
        msgs=messages(); encoded=self.template.encode_conversation(msgs)
        for i, span in enumerate(encoded['message_spans']):
            self.assertEqual(span['content_start'], span['start']+1)
            self.assertEqual(span['end'], span['content_end']+1)
            if span['role'] == 'assistant':
                prompt=self.template.encode_prompt(msgs[:i])
                self.assertEqual(prompt['input_ids'], encoded['input_ids'][:span['content_start']])
                self.assertEqual(prompt['response_start'],len(prompt['input_ids']))
                self.assertNotIn(c.EOS,prompt['input_ids'])

    def test_special_looking_user_content_is_never_a_role_boundary(self):
        text='<BOS><EOS><PAD><SYSTEM><USER><ASSISTANT><EOT> <사용자> <봇>'
        msgs=[{'role':'user','content':text},{'role':'assistant','content':text}]
        encoded=self.template.encode_conversation(msgs)
        for span in encoded['message_spans']:
            body=encoded['input_ids'][span['content_start']:span['content_end']]
            self.assertTrue(all(6 <= token < self.template.base_vocab_size for token in body))
        self.assertEqual(self.template.decode_conversation(encoded['input_ids']),msgs)

    def test_empty_whitespace_decomposed_hangul_emoji_and_line_separators(self):
        for text in ['', '  \t\n', '가 가 🙂𐍈\u2028\u0085\r\n끝 ', '\ufffd']:
            msgs=[{'role':'user','content':text},{'role':'assistant','content':text}]
            encoded=self.template.encode_conversation(msgs)
            self.assertEqual(self.template.decode_conversation(encoded['input_ids']),msgs)

    def test_invalid_roles_and_conversation_endings(self):
        cases=[[], [{'role':'system','content':'only system'}],
               [{'role':'assistant','content':'first'}],
               [{'role':'user','content':'a'},{'role':'user','content':'b'}],
               [{'role':'user','content':'a'},{'role':'system','content':'late'}],
               [{'role':'tool','content':'unsupported'}],
               [{'role':'user','content':None}],
               [{'role':'user','content':'\ud800'}],
               [{'role':'user','content':'a','extra':1}]]
        for msgs in cases:
            with self.subTest(messages=msgs), self.assertRaises(ValueError):
                self.template.encode_conversation(msgs)
        with self.assertRaises(ValueError): self.template.encode_prompt(messages())
        with self.assertRaises(ValueError): self.template.encode_conversation(messages()[:-1])

    def test_response_turn_and_conversation_end_are_distinct(self):
        raw=self.template.bpe.encode('안녕🙂')
        for suffix, stop, closed in [([self.template.eot_id],'turn_end',False),
                                    ([c.EOS],'conversation_end',True),
                                    ([self.template.eot_id,c.EOS],'conversation_end',True),
                                    ([], 'incomplete',False)]:
            self.assertEqual(self.template.decode_response(raw+suffix),
                             {'text':'안녕🙂','stop_reason':stop,'conversation_closed':closed})
        self.assertEqual(self.template.decode_response([self.template.eot_id])['text'],'')

    def test_response_rejects_role_injection_trailing_tokens_and_partial_utf8(self):
        for ids in [[3],[4],[5],[0],[c.BOS],[self.template.eot_id,100],
                    [c.EOS,self.template.eot_id],[self.template.eot_id]*2,
                    [self.template.vocab_size],[True],[-1], [0xe3+6,self.template.eot_id]]:
            with self.subTest(ids=ids),self.assertRaises(ValueError): self.template.decode_response(ids)

    def test_conversation_decoder_rejects_missing_and_injected_boundaries(self):
        ids=self.template.encode_conversation(messages())['input_ids']
        cases=[ids[1:], ids[:-1], ids+[c.EOS], [c.BOS,c.EOS], [c.BOS,3,100,c.EOS],
               [c.BOS,3,4,self.template.eot_id,c.EOS],
               [c.BOS,4,self.template.eot_id,c.EOS], [False]+ids[1:]]
        for tokens in cases:
            with self.subTest(ids=tokens),self.assertRaises(ValueError): self.template.decode_conversation(tokens)

    def test_template_save_reload_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'template.json'; self.template.save(path)
            restored=c.ChatTemplate.load(path)
            self.assertEqual(restored.encode_conversation(messages()),self.template.encode_conversation(messages()))
            original=json.loads(path.read_bytes())
            for key,value in [('eot_id',7),('vocab_size',768),('format','unknown'),('roles',{})]:
                path.write_bytes(c.json_bytes({**original,key:value}))
                with self.assertRaises(ValueError): c.ChatTemplate.load(path)

    def test_exported_template_metadata_does_not_alias_internal_state(self):
        original=self.template.encode_conversation(messages())
        metadata=self.template.to_dict()
        metadata['roles']['user']=99
        metadata['tokenizer']['merges'].clear()
        self.assertEqual(self.template.encode_conversation(messages()),original)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.output=self.root/'preview'

    def build(self, **kwargs):
        return p.prepare(p.DEFAULT_DATA,p.DEFAULT_TOKENIZER,self.output,allow_draft=True,**kwargs)

    def test_full_draft_roundtrips_and_all_response_prefixes(self):
        manifest=self.build()
        audit=p.module('chat051_verify',VERSION/'2.test/test.py').verify(self.output,allow_draft=True)
        self.assertEqual(audit['conversation_roundtrips'],48)
        self.assertEqual(audit['assistant_prefix_and_response_checks'],54)
        self.assertFalse(audit['training_ready'])
        self.assertEqual(sum(s['statistics']['exceeds_64_tokens'] for s in manifest['splits'].values()),48)

    def test_default_rejects_unreviewed_input_and_export(self):
        with self.assertRaisesRegex(ValueError,'draft'):
            p.prepare(p.DEFAULT_DATA,p.DEFAULT_TOKENIZER,self.output)
        self.assertFalse(self.output.exists())
        self.build()
        with self.assertRaisesRegex(ValueError,'draft'): p.load_prepared(self.output)

    def test_reviewed_fixture_keeps_receipts_and_split_ids(self):
        # Synthetic approval receipts are confined to this temporary test fixture.
        rows=p.data.read_jsonl(p.data.HERE/'examples.jsonl')
        reviews=[{'id':r['id'],'content_sha256':p.data.content_hash(r),'reviewer_kind':'human',
                  'reviewer':'synthetic-unit-test-not-real-review','decision':'approved',
                  'note':'Test fixture only.','checks':{k:True for k in p.data.CRITERIA}} for r in rows]
        receipts=self.root/'reviews.jsonl';receipts.write_bytes(p.data.jsonl(reviews))
        source=self.root/'reviewed'
        p.data.build(p.data.HERE/'examples.jsonl',p.data.HERE/'sources.json',receipts,source,mode='reviewed')
        p.prepare(source,p.DEFAULT_TOKENIZER,self.output)
        manifest,_,splits,encoded=p.load_prepared(self.output)
        self.assertTrue(manifest['training_ready'])
        for name in p.data.SPLITS:
            self.assertEqual([r['id'] for r in splits[name]],[e['id'] for e in encoded[name]])
            self.assertEqual((source/f'{name}.jsonl').read_bytes(),(self.output/'source'/f'{name}.jsonl').read_bytes())

    def test_length_limit_rejects_without_truncation_or_partial_export(self):
        with self.assertRaisesRegex(ValueError,'48 conversations exceed'): self.build(max_tokens=64)
        self.assertFalse(self.output.exists())
        self.build(max_tokens=222)
        before=(self.output/'train.jsonl').read_bytes()
        with self.assertRaisesRegex(ValueError,'exceed'): self.build(max_tokens=221)
        self.assertEqual(before,(self.output/'train.jsonl').read_bytes())

    def test_same_export_is_idempotent_and_different_output_is_protected(self):
        self.build(); before={str(f.relative_to(self.output)):f.read_bytes() for f in self.output.rglob('*') if f.is_file()}
        self.build()
        self.assertEqual(before,{str(f.relative_to(self.output)):f.read_bytes() for f in self.output.rglob('*') if f.is_file()})
        (self.output/'train.jsonl').write_text('user content')
        with self.assertRaises(FileExistsError): self.build()
        self.assertEqual((self.output/'train.jsonl').read_text(),'user content')

    def test_source_paths_cannot_be_overwritten(self):
        for target in [p.DEFAULT_DATA,p.DEFAULT_DATA/'nested',p.DEFAULT_DATA.parent,p.DEFAULT_TOKENIZER.parent]:
            with self.subTest(path=target),self.assertRaisesRegex(ValueError,'separate'):
                p.prepare(p.DEFAULT_DATA,p.DEFAULT_TOKENIZER,target,allow_draft=True)

    def test_mutated_ids_or_spans_rejected_even_with_updated_file_hash(self):
        for field in ['input_ids','message_spans']:
            target=self.root/field
            p.prepare(p.DEFAULT_DATA,p.DEFAULT_TOKENIZER,target,allow_draft=True)
            rows=p.data.read_jsonl(target/'train.jsonl')
            if field=='input_ids': rows[0][field][2]=100
            else: rows[0][field][0]['content_end']+=1
            raw=p.data.jsonl(rows); (target/'train.jsonl').write_bytes(raw)
            manifest=json.loads((target/'manifest.json').read_bytes())
            manifest['splits']['train']['sha256']=p.data.digest(raw)
            (target/'manifest.json').write_bytes(p.data.encode(manifest))
            with self.assertRaisesRegex(ValueError,'differs'): p.load_prepared(target,for_training=False)

    def test_readiness_and_tokenizer_tamper_rejected(self):
        self.build(); path=self.output/'manifest.json'; original=json.loads(path.read_bytes())
        path.write_bytes(p.data.encode({**original,'training_ready':True}))
        with self.assertRaisesRegex(ValueError,'readiness'): p.load_prepared(self.output,for_training=False)
        path.write_bytes(p.data.encode(original)); (self.output/'template.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'template hash'): p.load_prepared(self.output,for_training=False)

    def test_manifest_vocabulary_and_length_contract_checked(self):
        self.build(); path=self.output/'manifest.json'; original=json.loads(path.read_bytes())
        for key,value in [('vocab_size',768),('eot_id',7),('base_tokenizer_sha256','wrong'),
                          ('max_tokens_limit',64),('max_tokens_limit',True)]:
            path.write_bytes(p.data.encode({**original,key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):
                p.load_prepared(self.output,for_training=False)

    def test_cli_prepare_verify_and_draft_refusal(self):
        cmd=[sys.executable,str(VERSION/'1.train/prepare.py'),'--output-dir',str(self.output)]
        self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
        subprocess.run(cmd+['--allow-draft'],check=True,capture_output=True)
        out=subprocess.run([sys.executable,str(VERSION/'2.test/test.py'),'--dataset',str(self.output),
                            '--allow-draft'],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(out.stdout)['conversation_roundtrips'],48)


if __name__ == '__main__':
    unittest.main()
