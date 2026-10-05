"""Release provenance, review boundaries and actual network compatibility."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

VERSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION/'1.train'))
import release as r


class ReleaseTests(unittest.TestCase):
    def test_release_rebuild_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'release'
            first = r.build(output)
            self.assertEqual(first, r.build(output))
            self.assertEqual(first, r.verify(output))
            self.assertTrue(first['preparation_complete'])
            self.assertTrue(first['training_ready'])
            self.assertTrue(first['human_approval'])
            self.assertEqual(first['human_approved_conversations'], 48)
            self.assertLessEqual(first['max_conversation_tokens'], 256)
            r.load_prepared(output/'prepared')
            (output/'unexpected.txt').write_text('keep user file')
            with self.assertRaises(FileExistsError):
                r.build(output)
            self.assertEqual((output/'unexpected.txt').read_text(), 'keep user file')

    def test_release_requires_complete_current_human_approvals(self):
        rows = r.data.read_jsonl(r.SOURCE/'examples.jsonl')
        original = r.data.read_jsonl(r.SOURCE/'reviews.jsonl')
        r.validate_approvals(rows, original)
        variants = [original[:-1], original + [original[0]]]
        for field, value in [('reviewer_kind', 'ai'), ('decision', 'rejected'),
                             ('content_sha256', 'bad')]:
            q = copy.deepcopy(original); q[0][field] = value; variants.append(q)
        q = copy.deepcopy(original); q[0]['checks']['correctness'] = False; variants.append(q)
        for q in variants:
            with self.assertRaises(ValueError):
                r.validate_approvals(rows, q)

    def test_modified_model_config_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'release'; r.build(output)
            config = json.loads((output/'model_config.json').read_bytes())
            config['block_size'] = 64
            (output/'model_config.json').write_bytes(r.data.encode(config))
            with self.assertRaises(ValueError):
                r.verify(output)

    def test_input_change_cannot_reuse_quality_review(self):
        rows = r.data.read_jsonl(r.SOURCE/'examples.jsonl')
        original = r.data.read_jsonl(r.SOURCE/'reviews.jsonl')
        rows[0]['messages'][-1]['content'] = 'Changed answer'
        with self.assertRaises(ValueError):
            r.validate_approvals(rows, original)

    def test_missing_approval_build_leaves_no_partial_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'inputs'; source.mkdir()
            for name in ('examples.jsonl', 'sources.json', 'reviews.jsonl'):
                (source/name).write_bytes((r.SOURCE/name).read_bytes())
            reviews = r.data.read_jsonl(source/'reviews.jsonl')
            (source/'reviews.jsonl').write_bytes(r.data.jsonl(reviews[:-1]))
            output = Path(tmp)/'output'
            with patch.object(r, 'SOURCE', source), self.assertRaises(ValueError):
                r.build(output)
            self.assertFalse(output.exists())

    def test_approval_tamper_in_archived_release_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'release'; r.build(output)
            path = output/'inputs/reviews.jsonl'
            reviews = r.data.read_jsonl(path)
            reviews[0]['decision'] = 'rejected'
            path.write_bytes(r.data.jsonl(reviews))
            with self.assertRaises(ValueError):
                r.verify(output)

    def test_all_data_forward_eot_gradient_and_causality(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(1234)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'release'; r.build(output)
            _, template, _, splits = r.load_prepared(output/'prepared', for_training=False)
            config = json.loads((output/'model_config.json').read_bytes())
            net = r.setup.build_network(config, template).eval()
            self.assertIs(net.embedding.weight, net.head.weight)
            with torch.no_grad():
                for records in splits.values():
                    for row in records:
                        x = torch.tensor([row['input_ids']])
                        logits = net(x)
                        self.assertEqual(logits.shape, (1, len(row['input_ids']), 769))
                        self.assertTrue(torch.isfinite(logits[..., [1] + list(range(6, 769))]).all())
                        self.assertTrue(torch.isneginf(logits[..., [0, 2, 3, 4, 5]]).all())
                x = torch.tensor([splits['train'][0]['input_ids']])
                changed = x.clone(); changed[0, -1] = 10
                torch.testing.assert_close(net(x)[:, :-1], net(changed)[:, :-1], rtol=0, atol=0)
            # One diagnostic backward pass, no optimizer step or training claim.
            loss = torch.nn.functional.cross_entropy(net(x)[:, -3], torch.tensor([template.eot_id]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
            self.assertGreater(net.embedding.weight.grad[template.eot_id].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
