"""SFT provenance, Unicode, split isolation and explicit review release contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('sft050_tests', ROOT/'data/sft/v0.5.0/dataset.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
SOURCES = json.loads((d.HERE/'sources.json').read_bytes())


def fixture(count=9):
    return [{'id': f'fixture-{i}', 'task': 'format', 'template_id': f'template-{i}',
             'source': 'original-ai-assisted-v1', 'rationale': 'Synthetic test fixture only.',
             'messages': [{'role': 'user', 'content': f'요청 {i}'},
                          {'role': 'assistant', 'content': f'응답 {i}'}]} for i in range(count)]


def approval(row):
    return {'id': row['id'], 'content_sha256': d.content_hash(row),
            'reviewer_kind': 'human', 'reviewer': 'synthetic-test-fixture-not-a-real-review',
            'decision': 'approved', 'note': 'Unit test receipt only.',
            'checks': {k: True for k in d.CRITERIA}}


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root/'examples.jsonl'
        self.sources = self.root/'sources.json'
        self.reviews = self.root/'reviews.jsonl'
        self.output = self.root/'bundle'
        self.sources.write_bytes(d.encode(SOURCES))
        self.reviews.write_bytes(b'')

    def build(self, rows=None, reviews=(), mode='draft', output=None):
        self.input.write_bytes(d.jsonl(fixture() if rows is None else rows))
        self.reviews.write_bytes(d.jsonl(reviews))
        return d.build(self.input, self.sources, self.reviews, output or self.output, mode=mode)

    def rewrite_split(self, name, entries):
        path = self.output/f'{name}.jsonl'; path.write_bytes(d.jsonl(entries))
        mf = self.output/'manifest.json'; manifest = json.loads(mf.read_bytes())
        manifest['splits'][name]['sha256'] = d.digest(path.read_bytes())
        mf.write_bytes(d.encode(manifest))

    def test_shipped_examples_are_reproducible_and_cover_all_tasks(self):
        rows = d.read_jsonl(d.HERE/'examples.jsonl')
        manifest = self.build(rows)
        self.assertEqual(len(rows), 48)
        self.assertEqual(sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows), 6)
        self.assertFalse(manifest['training_ready'])
        for name in d.SPLITS:
            self.assertEqual(manifest['splits'][name]['conversations'], 16)
            self.assertEqual(set(manifest['splits'][name]['tasks']), set(d.TASKS))
            self.assertEqual((self.output/f'{name}.jsonl').read_bytes(), (d.HERE/'draft'/f'{name}.jsonl').read_bytes())
        other = self.root/'other'
        self.build(list(reversed(rows)), output=other)
        for name in d.SPLITS:
            self.assertEqual((self.output/f'{name}.jsonl').read_bytes(), (other/f'{name}.jsonl').read_bytes())

    def test_unicode_whitespace_role_literals_and_jsonl_separators_roundtrip(self):
        rows = fixture()
        rows[0]['messages'][1]['content'] = '  가🙂 <사용자> <봇>\n\t끝\u2028다음\u0085줄  '
        self.build(rows)
        _, splits = d.load_bundle(self.output, for_training=False)
        actual = next(r for s in splits.values() for r in s if r['id']==rows[0]['id'])
        self.assertEqual(actual, rows[0])

    def test_union_before_dedup_preserves_template_bridge(self):
        rows = fixture()
        duplicate = copy.deepcopy(rows[0]); duplicate['id']='fixture-duplicate'
        duplicate['template_id']=rows[1]['template_id']
        rows.append(duplicate)
        manifest = self.build(rows)
        self.assertEqual(len(manifest['removed_duplicates']), 1)
        membership = {e['record']['id']: name for name in d.SPLITS
                      for e in d.read_jsonl(self.output/f'{name}.jsonl')}
        self.assertEqual(membership['fixture-0'], membership['fixture-1'])

    def test_followup_user_turns_stay_together(self):
        rows = fixture()
        for i in (0, 1):
            rows[i]['messages'] += [{'role':'user','content':'같은 후속 지시'},
                                     {'role':'assistant','content':'후속 응답'}]
        owners = d.assign_groups(rows, 1234)
        self.assertEqual(owners[rows[0]['id']], owners[rows[1]['id']])

    def test_normalized_duplicate_and_prompt_matching(self):
        rows = fixture()
        rows[0]['messages'][0]['content']='가  나'
        rows[1]['messages'][0]['content']='가\t나'
        owners=d.assign_groups(rows, 1234)
        self.assertEqual(owners[rows[0]['id']], owners[rows[1]['id']])

    def test_draft_cannot_be_loaded_for_training_or_promoted_by_manifest_only(self):
        self.build()
        with self.assertRaisesRegex(ValueError, 'draft'):
            d.load_bundle(self.output)
        path=self.output/'manifest.json'; manifest=json.loads(path.read_bytes())
        manifest.update(mode='reviewed',training_ready=True); path.write_bytes(d.encode(manifest))
        with self.assertRaisesRegex(ValueError, 'receipt'):
            d.load_bundle(self.output)

    def test_reviewed_export_and_review_filter_keep_original_split(self):
        rows=fixture(); draft=self.build(rows)
        owners={e['record']['id']: name for name in d.SPLITS for e in d.read_jsonl(self.output/f'{name}.jsonl')}
        receipts=[approval(r) for r in rows]
        receipts[0]['decision']='rejected'
        self.build(rows, receipts, 'reviewed')
        manifest,splits=d.load_bundle(self.output)
        self.assertTrue(manifest['training_ready'])
        self.assertIn(rows[0]['id'],manifest['excluded_ids'])
        for name, records in splits.items():
            self.assertTrue(all(owners[r['id']]==name for r in records))
            self.assertNotIn(rows[0]['id'],[r['id'] for r in records])

    def test_stale_review_rejected_before_existing_export_is_touched(self):
        rows=fixture(); self.build(rows)
        before={p.name:p.read_bytes() for p in self.output.iterdir()}
        receipts=[approval(r) for r in rows]
        rows[0]['messages'][1]['content']='edited answer'
        with self.assertRaisesRegex(ValueError,'stale review'): self.build(rows,receipts,'reviewed')
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.output.iterdir()})

    def test_ai_incomplete_duplicate_and_unknown_reviews_are_rejected(self):
        rows=fixture(); good=approval(rows[0])
        variants=[{**good,'reviewer_kind':'ai'}, {**good,'checks':{}}, {**good,'reviewer':''},
                  {**good,'id':'unknown'}, {**good,'checks':{k:1 for k in d.CRITERIA}}]
        for bad in variants:
            with self.subTest(bad=bad),self.assertRaises(ValueError): d.review_index(rows,[bad])
        with self.assertRaises(ValueError): d.review_index(rows,[good,good])

    def test_reviewed_build_without_reviews_never_creates_output(self):
        with self.assertRaisesRegex(ValueError,'eligible'): self.build(mode='reviewed')
        self.assertFalse(self.output.exists())

    def test_roles_empty_text_controls_unknown_sources_and_ids(self):
        for change in [lambda r:r.update(source='unknown'), lambda r:r.update(task='unknown'),
                       lambda r:r.update(messages=[]),
                       lambda r:r['messages'][1].update(role='user'),
                       lambda r:r['messages'][1].update(content=' '),
                       lambda r:r['messages'][1].update(content='\ufffd'),
                       lambda r:r['messages'][1].update(content='bad\x00'),
                       lambda r:r['messages'][1].update(content='\ud800'),
                       lambda r:r['messages'].append({'role':'system','content':'late system'})]:
            rows=fixture(); change(rows[0])
            with self.assertRaises(ValueError): d.validate_records(rows,SOURCES)
        rows=fixture(); rows[1]['id']=rows[0]['id']
        with self.assertRaises(ValueError): d.validate_records(rows,SOURCES)

    def test_tamper_hash_and_statistics_detected(self):
        self.build(); path=self.output/'train.jsonl'
        path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'hash mismatch'): d.load_bundle(self.output,for_training=False)
        self.build(); path=self.output/'manifest.json'; manifest=json.loads(path.read_bytes())
        manifest['splits']['valid']['conversations']+=1; path.write_bytes(d.encode(manifest))
        with self.assertRaisesRegex(ValueError,'statistics'): d.load_bundle(self.output,for_training=False)

    def test_leakage_detected_even_if_file_and_content_hashes_updated(self):
        self.build()
        train=d.read_jsonl(self.output/'train.jsonl'); valid=d.read_jsonl(self.output/'valid.jsonl')
        valid[0]['record']['messages'][0]['content']=train[0]['record']['messages'][0]['content']
        valid[0]['content_sha256']=d.content_hash(valid[0]['record'])
        self.rewrite_split('valid',valid)
        with self.assertRaisesRegex(ValueError,'leakage'): d.load_bundle(self.output,for_training=False)

    def test_insufficient_groups_refuse_leaky_fallback(self):
        rows=fixture()
        for row in rows: row['template_id']='one-template'
        with self.assertRaisesRegex(ValueError,'independent'): self.build(rows)
        self.assertFalse(self.output.exists())

    def test_source_input_overwrite_is_rejected(self):
        rows=fixture(); self.input=self.root/'train.jsonl'
        self.input.write_bytes(d.jsonl(rows)); self.reviews.write_bytes(b'')
        before=self.input.read_bytes()
        with self.assertRaisesRegex(ValueError,'overwrite'):
            d.build(self.input,self.sources,self.reviews,self.root)
        self.assertEqual(before,self.input.read_bytes())

    def test_legacy_audit_checks_later_turns_and_flags_malformed_lines(self):
        (self.root/'train.txt').write_text('<사용자> 시작 <봇> 답 <사용자> 공통 <봇> 끝\n<사용자> 답 없음\n')
        (self.root/'valid.txt').write_text('<사용자> 다른 시작 <봇> 답 <사용자> 공통 <봇> 다른 끝\n')
        report=d.legacy_audit(self.root)
        self.assertEqual(report['shared_user_turns'],1)
        self.assertEqual(report['train']['malformed_lines'],[2])

    def test_prepare_and_audit_cli(self):
        subprocess.run([sys.executable,str(ROOT/'llm/v0.5/v0.5.0/1.train/prepare.py'),
                        '--output-dir',str(self.output)],check=True,capture_output=True)
        cmd=[sys.executable,str(Path(__file__).with_name('test.py')),'--dataset',str(self.output)]
        result=subprocess.run(cmd,check=True,capture_output=True,text=True)
        self.assertFalse(json.loads(result.stdout)['training_ready'])
        self.assertNotEqual(subprocess.run(cmd+['--require-reviewed'],capture_output=True).returncode,0)

    def test_review_cli_requires_all_checks_and_binds_content(self):
        rows=fixture(); self.input.write_bytes(d.jsonl(rows))
        cmd=[sys.executable,str(d.HERE/'review.py'),'--input',str(self.input),
             '--reviews',str(self.reviews),'--id',rows[0]['id'],'--decision','approved',
             '--reviewer','synthetic-test-only','--note','Fixture, not real human review.']
        self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
        self.assertEqual(self.reviews.read_bytes(),b'')
        subprocess.run(cmd+['--checked',*d.CRITERIA],check=True,capture_output=True)
        receipts=d.read_jsonl(self.reviews)
        self.assertEqual(receipts[0]['content_sha256'],d.content_hash(rows[0]))


if __name__ == '__main__':
    unittest.main()
