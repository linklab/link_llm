"""학습 의미를 검증하는 CPU 테스트. 별도 데이터나 저장 모델 불필요."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v030_test", VERSION / "0.model/lm.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Invariants(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.lm = m.Model()
        self.lm.DEVICE = "cpu"
        self.lm.BLOCK_SIZE = 3
        self.lm.EMBED = 16
        self.sentences = ["a b c d e f", "b c", "a"]
        self.lm.initialize(self.sentences)

    def test_future_cannot_change_past(self):
        net = self.lm.net
        a = torch.tensor([[1, 2, 3]])
        b = torch.tensor([[1, 2, 4]])
        torch.testing.assert_close(net(a)[:, :2], net(b)[:, :2], rtol=0, atol=0)

    def test_padding_and_all_pad_are_finite(self):
        net = self.lm.net
        short = net(torch.tensor([[1, 2]]))
        padded = net(torch.tensor([[1, 2, 0]]))
        torch.testing.assert_close(short, padded[:, :2])
        all_pad = net(torch.zeros(1, 3, dtype=torch.long))
        self.assertTrue(torch.isfinite(all_pad).all())
        all_pad.sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_shift_coverage_and_context(self):
        rows = self.lm.make_windows(self.sentences)
        actual = []
        for x, y, _ in rows:
            for j, target in enumerate(y):
                if target != -100:
                    actual.append((x[:j + 1], target))
        expected = []
        for sentence in self.sentences:
            ids = [self.lm.stoi[t] for t in self.lm.prepare(self.lm.tokenize(sentence))]
            for i in range(1, len(ids)):
                expected.append((ids[max(0, i - 3):i], ids[i]))
        self.assertEqual(actual, expected)

    def test_ignored_targets_have_zero_gradient(self):
        logits = torch.randn(2, 3, 8, requires_grad=True)
        targets = torch.tensor([[1, 2, -100], [3, -100, -100]])
        loss = m.sequence_loss(logits, targets)
        reference = torch.nn.functional.cross_entropy(logits[targets != -100], targets[targets != -100])
        torch.testing.assert_close(loss, reference)
        loss.backward()
        self.assertEqual(logits.grad[targets == -100].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            m.sequence_loss(logits, torch.full_like(targets, -100))

    def test_batch_evaluation_matches_token_prob_including_oov(self):
        sentences = self.sentences + ["a unknown c d e f", "", "a b"]
        logs = []
        for sentence in sentences:
            tokens = self.lm.prepare(self.lm.tokenize(sentence))
            logs.extend(math.log(self.lm.token_prob(tokens[:i], tokens[i]))
                        for i in range(1, len(tokens)))
        self.lm.net.train()
        result = self.lm.evaluate(sentences)
        self.assertAlmostEqual(result["ppl"], math.exp(-sum(logs) / len(logs)), places=5)
        self.assertEqual(result["n"], len(logs))
        self.assertTrue(self.lm.net.training)
        self.assertLess(result["coverage"], 1)
        self.lm.EVAL_BATCH = 1
        self.assertAlmostEqual(result["ppl"], self.lm.perplexity(sentences), places=5)

    def test_gradient_reaches_qkv(self):
        x, y, _ = m._data.collate(self.lm.make_windows(self.sentences)[:2])
        m.sequence_loss(self.lm.net(x), y).backward()
        for name in ("embedding", "query", "key", "value", "head"):
            grad = getattr(self.lm.net, name).weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)

    def test_save_load_restores_nondefault_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            self.lm.save(path)
            restored = m.Model().load(path)
            self.assertEqual(restored.BLOCK_SIZE, 3)
            self.assertEqual(restored.EMBED, 16)
            x = torch.tensor([[1, 2, 3]])
            torch.testing.assert_close(self.lm.net(x), restored.net(x), rtol=0, atol=0)
            self.assertEqual(self.lm.generate("a"), restored.generate("a"))

    def test_end_empty_unknown_and_sampling(self):
        self.assertEqual(self.lm.generate(""), "")
        self.assertEqual(self.lm.generate("unknown"), "unknown")
        with patch.object(self.lm, "next_token", return_value=self.lm.END):
            self.assertEqual(self.lm.generate("a"), "a")
        dist = self.lm.next_dist(["a"])
        self.assertNotIn(self.lm.PAD, dist)
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
        self.lm._top_k = 1
        self.lm._top_p = 1.0
        self.assertEqual(self.lm.next_token(["a"], 0.011), max(dist, key=dist.get))

    def test_invalid_shape_and_reserved_pad(self):
        for x in (torch.ones(1, 4, dtype=torch.long), torch.ones(3, dtype=torch.long)):
            with self.assertRaises(ValueError):
                self.lm.net(x)
        with self.assertRaises(ValueError):
            self.lm.build_vocab(["a <PAD> b"])
        self.assertEqual(self.lm.evaluate([])["n"], 0)

    def test_small_batch_can_overfit(self):
        self.lm.EPOCHS = 120
        self.lm.LR = 0.03
        self.lm.train(["a b c"] * 4)
        result = self.lm.evaluate(["a b c"])
        self.assertLess(result["ppl"], 1.05)
        self.assertEqual(result["top1"], 1.0)

    def test_early_stopping_restores_best_validation_weights(self):
        self.lm.EPOCHS = 10
        self.lm.PATIENCE = 2
        states = []
        scores = iter([9.0, 8.0, 10.0, 11.0])

        def validation(_):
            states.append({k: v.detach().clone() for k, v in self.lm.net.state_dict().items()})
            return {"ppl": next(scores)}

        with patch.object(self.lm, "evaluate_rows", side_effect=validation):
            self.lm.train(["a b c"], ["a b"])
        self.assertEqual(len(self.lm.history), 4)
        self.assertEqual(self.lm.best_epoch, 2)
        for key, value in self.lm.net.state_dict().items():
            torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
