"""Release provenance, review boundaries and actual network compatibility."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

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
            self.assertFalse(first['training_ready'])
            self.assertFalse(first['human_approval'])
            self.assertEqual(first['ai_reviewed_conversations'], 48)
            self.assertLessEqual(first['max_conversation_tokens'], 256)
            with self.assertRaises(ValueError):
                r.load_prepared(output/'prepared')
            (output/'unexpected.txt').write_text('keep user file')
            with self.assertRaises(FileExistsError):
                r.build(output)
            self.assertEqual((output/'unexpected.txt').read_text(), 'keep user file')

    def test_quality_requires_current_complete_ai_receipts(self):
        rows = r.data.read_jsonl(r.SOURCE/'examples.jsonl')
        original = json.loads((r.SOURCE/'ai_quality_audit.json').read_bytes())
        variants = []
        for field, value in [('reviewer_kind', 'human'), ('human_approval', True)]:
            q = copy.deepcopy(original); q[field] = value; variants.append(q)
        q = copy.deepcopy(original); q['records'].pop(); variants.append(q)
        q = copy.deepcopy(original); q['records'][0]['content_sha256'] = 'bad'; variants.append(q)
        q = copy.deepcopy(original); q['records'][0]['assistant_responses_checked'] = 0; variants.append(q)
        for q in variants:
            with self.assertRaises(ValueError):
                r.validate_quality(rows, q)

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
        original = json.loads((r.SOURCE/'ai_quality_audit.json').read_bytes())
        rows[0]['messages'][-1]['content'] = 'Changed answer'
        with self.assertRaises(ValueError):
            r.validate_quality(rows, original)

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
