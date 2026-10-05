import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch

V = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p052test', V/'1.train/prepare.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
s = p.sft


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        _, self.t, self.splits, _ = p.p.load_prepared(p.DEFAULT_SOURCE)
        self.r = {'id': 'fixture', 'messages': [
            {'role': 'system', 'content': '짧게'},
            {'role': 'user', 'content': '첫 질문'},
            {'role': 'assistant', 'content': '첫 응답'},
            {'role': 'user', 'content': '다음 질문'},
            {'role': 'assistant', 'content': '🙂 끝'}]}

    def test_each_assistant_once_with_eot_and_only_final_eos(self):
        rows = s.examples(self.t, self.r)
        self.assertEqual(len(rows), 2)
        for i, row in enumerate(rows):
            body = self.t.bpe.encode(self.r['messages'][2+i*2]['content'])
            expected = body + [self.t.eot_id] + ([1] if i == 1 else [])
            self.assertEqual([y for y in row['labels'] if y != -100], expected)
            first = next(j for j,y in enumerate(row['labels']) if y != -100)
            self.assertEqual(row['input_ids'][first], 4)
            prompt = self.t.encode_prompt(self.r['messages'][:2+i*2])['input_ids']
            self.assertEqual(row['input_ids'][:first+1], prompt)
            self.assertTrue(all(y == -100 for y in row['labels'][:first]))

    def test_empty_response_still_trains_terminators(self):
        self.r['messages'][-1]['content'] = ''
        last = s.examples(self.t, self.r)[-1]
        self.assertEqual([y for y in last['labels'] if y != -100], [self.t.eot_id, 1])

    def test_literals_are_not_role_targets(self):
        self.r['messages'][-1]['content'] = '<ASSISTANT>  가🙂\n'
        row = s.examples(self.t, self.r)[-1]
        ids = [y for y in row['labels'] if y != -100][:-2]
        self.assertEqual(self.t.bpe.decode(ids), self.r['messages'][-1]['content'])
        self.assertTrue(all(i >= 6 for i in ids))

    def test_context_exact_boundary_and_rejection(self):
        maximum = max(len(r['input_ids']) for r in s.examples(self.t, self.r))
        s.examples(self.t, self.r, maximum)
        with self.assertRaises(ValueError): s.examples(self.t, self.r, maximum-1)
        for size in [0, True, 1.2]:
            with self.assertRaises(ValueError): s.examples(self.t, self.r, size)

    def test_drop_pairs_preserves_system_current_user_and_response(self):
        original = s.examples(self.t, self.r)
        minimum = max(len(original[0]['input_ids']), len(self.t.encode_conversation(
            [self.r['messages'][0]] + self.r['messages'][3:])['input_ids'])-1)
        rows = s.examples(self.t, self.r, minimum, 'drop-oldest-pairs')
        self.assertEqual(rows[-1]['kept_message_indices'], [0,3,4])
        self.assertEqual(rows[-1]['dropped_message_indices'], [1,2])
        self.assertEqual([y for y in rows[-1]['labels'] if y != -100],
                         [y for y in original[-1]['labels'] if y != -100])

    def test_oversized_response_user_or_system_never_sliced(self):
        for index in [0, 1, 2]:
            r = copy.deepcopy(self.r); r['messages'][index]['content'] = 'x'*500
            with self.assertRaises(ValueError): s.examples(self.t, r, 32, 'drop-oldest-pairs')

    def test_no_assistant_rejected_and_no_system_pair_removal(self):
        with self.assertRaises(ValueError):
            s.examples(self.t, {'id':'empty','messages':self.r['messages'][:2]})
        self.r['messages'] = self.r['messages'][1:]
        rows = s.examples(self.t,self.r)
        limit = max(len(rows[0]['input_ids']),len(self.t.encode_conversation(self.r['messages'][2:])['input_ids'])-1)
        reduced = s.examples(self.t,self.r,limit,'drop-oldest-pairs')
        self.assertEqual(reduced[-1]['kept_message_indices'],[2,3])
        self.assertEqual(reduced[-1]['dropped_message_indices'],[0,1])

    def test_padding_mask_and_loss_denominator(self):
        batch = s.collate(s.examples(self.t, self.r))
        for ids, labels, mask in zip(batch['input_ids'],batch['labels'],batch['attention_mask']):
            for i in range(len(ids)):
                if not mask[i]: self.assertEqual((ids[i],labels[i]), (0,-100))
        labels = torch.tensor(batch['labels'])
        torch.manual_seed(9)
        logits = torch.randn(*labels.shape, self.t.vocab_size, requires_grad=True)
        loss,n = s.response_loss(logits,labels)
        wanted = torch.stack([-torch.log_softmax(logits[b,t],-1)[labels[b,t]]
                              for b,t in labels.ne(-100).nonzero()]).mean()
        torch.testing.assert_close(loss,wanted)
        self.assertEqual(n,int(labels.ne(-100).sum()))
        loss.backward()
        self.assertEqual(float(logits.grad[labels.eq(-100)].abs().sum()),0)
        self.assertGreater(float(logits.grad[labels.ne(-100)].abs().sum()),0)

    def test_no_targets_zero_loss_with_masked_logits(self):
        logits = torch.zeros(2,3,10,requires_grad=True)
        masked = logits.masked_fill(torch.arange(10).eq(0),float('-inf'))
        loss,n = s.response_loss(masked,torch.full((2,3),-100))
        self.assertEqual(n,0); self.assertEqual(loss.item(),0)
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError): s.collate([])

    def test_export_determinism_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'; manifest = p.build(output=out)
            self.assertEqual(manifest,p.build(output=out))
            self.assertEqual(sum(x['examples'] for x in manifest['splits'].values()),54)
            path = out/'train.jsonl'; rows = p.p.data.read_jsonl(path)
            rows[0]['labels'][0] = 1
            path.write_bytes(p.p.data.jsonl(rows))
            manifest['splits']['train']['sha256'] = p.p.data.digest(path.read_bytes())
            (out/'manifest.json').write_bytes(p.p.data.encode(manifest))
            with self.assertRaises(ValueError): p.load(out)
            with self.assertRaises(FileExistsError): p.build(output=out)

    def test_draft_and_overflow_leave_no_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'out'
            with self.assertRaises(ValueError): p.build(output=out,context=10)
            self.assertFalse(out.exists())
            with self.assertRaises(ValueError): p.build(p.PREVIOUS/'1.train/preview',out)
            self.assertFalse(out.exists())

    def test_real_model_loss_gradient_and_batch_padding(self):
        setup = p.module('setup052',p.PREVIOUS/'0.model/model_setup.py')
        net = setup.build_network(setup.configuration(self.t),self.t).eval()
        rows = [e for r in self.splits['train'] for e in s.examples(self.t,r)]
        batch = s.collate(rows)
        x,y = torch.tensor(batch['input_ids']),torch.tensor(batch['labels'])
        logits = net(x)
        loss,n = s.response_loss(logits,y)
        self.assertTrue(torch.isfinite(loss));self.assertGreater(n,0)
        loss.backward()
        self.assertTrue(all(t.grad is None or torch.isfinite(t.grad).all() for t in net.parameters()))
        with torch.no_grad():
            for i,row in enumerate(rows[:2]):
                single = net(torch.tensor([row['input_ids']]))
                torch.testing.assert_close(logits[i,:len(row['input_ids'])],single[0],rtol=1e-5,atol=1e-6)


if __name__ == '__main__': unittest.main(verbosity=2)
