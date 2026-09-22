"""학습 의미를 검증하는 CPU 테스트. 별도 데이터나 저장 모델 불필요."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

VERSION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v034_test", VERSION / "0.model/lm.py")
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
        self.lm.net.eval()

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
        for name, parameter in self.lm.net.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_save_load_restores_nondefault_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            self.lm.save(path)
            # setUp의 원본과 같은 CPU에서 정확히 비교합니다. 기본 auto로
            # 로드하면 Apple Silicon에서는 MPS 모델에 CPU 입력을 주게 됩니다.
            restored = m.Model()
            restored.DEVICE = "cpu"
            # MPS 사용 가능 환경에서도 명시한 CPU 설정을 지키는지 검증합니다.
            with patch.object(torch.backends.mps, "is_available", return_value=True):
                restored.load(path)
            self.assertEqual(restored.device(), torch.device("cpu"))
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

    def test_long_prompt_generation_uses_bounded_context(self):
        seen = []
        def capture(module, args):
            seen.append(args[0].shape[1])
        handle = self.lm.net.register_forward_pre_hook(capture)
        self.lm.MAX_LENGTH = 4
        try:
            with torch.no_grad():
                self.lm.net.head.weight.zero_()
                self.lm.net.head.bias.zero_()
                self.lm.net.head.bias[self.lm.stoi["a"]] = 10
            output = self.lm.generate("a b c d e f")
            self.assertEqual(len(self.lm.tokenize(output)), 10)
            self.assertEqual(seen, [3] * 4)
            with torch.no_grad():
                self.lm.net.head.bias[self.lm.stoi[self.lm.END]] = 20
            seen.clear()
            self.lm.generate("a b c d e f")
            self.assertEqual(seen, [3])
        finally:
            handle.remove()

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



    def test_position_boundaries_order_and_gradient(self):
        net = self.lm.net
        x = torch.tensor([[1, 2, 3]])
        reversed_prefix = torch.tensor([[2, 1, 3]])
        self.assertFalse(torch.allclose(net(x)[:, -1, 1:], net(reversed_prefix)[:, -1, 1:]))
        self.assertEqual(net(torch.tensor([[1]])).shape[1], 1)
        with self.assertRaises(ValueError):
            net(torch.empty(1, 0, dtype=torch.long))
        with self.assertRaises(ValueError):
            net(torch.ones(1, 4, dtype=torch.long))
        m.sequence_loss(net(x), torch.tensor([[2, 3, 4]])).backward()
        for row in net.position.weight.grad:
            self.assertGreater(row.abs().sum().item(), 0)

    def test_prenorm_residual_and_ffn_paths(self):
        net = self.lm.net
        x = torch.tensor([[1, 2, 3]])
        h = net.embed_tokens(x)
        for block in net.blocks:
            after_attention = h + block.attention(block.ln1(h), x)
            expected = after_attention + block.ffn(block.ln2(after_attention))
            torch.testing.assert_close(block(h, x), expected)
            with torch.no_grad():
                block.projection.weight.zero_()
                block.ffn[2].weight.zero_()
                block.ffn[2].bias.zero_()
            torch.testing.assert_close(block(h, x), h, rtol=0, atol=0)

    def test_depth_initialization_and_validation(self):
        net = m.TransformerLM(100, 128, 8, 4, 256, 0.0, 3)
        self.assertEqual(len(net.blocks), 3)
        self.assertNotEqual(net.blocks[0].query.weight.data_ptr(),
                            net.blocks[1].query.weight.data_ptr())
        for block in net.blocks:
            self.assertAlmostEqual(block.query.weight.std().item(), 0.02, delta=0.001)
            for weight in (block.projection.weight, block.ffn[2].weight):
                self.assertAlmostEqual(weight.std().item(), 0.02 / math.sqrt(6), delta=0.0005)
            self.assertEqual(block.ffn[2].bias.abs().sum().item(), 0)
        self.assertEqual(net.embedding.weight[0].abs().sum().item(), 0)
        for layers in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                m.TransformerLM(10, 16, 3, 4, layers=layers)
        self.assertEqual(m.TransformerLM(10, 16, 3, 4, layers=1)(
            torch.ones(1, 3, dtype=torch.long)).shape, (1, 3, 10))

    def test_dropout_eval_and_config_roundtrip(self):
        self.lm.LAYERS = 2
        self.lm.HEADS = 2
        self.lm.FFN_HIDDEN = 24
        self.lm.DROPOUT = 0.5
        self.lm.initialize(self.sentences)
        net = self.lm.net
        x = torch.tensor([[1, 2, 3]])
        net.train()
        self.assertFalse(torch.equal(net(x), net(x)))
        self.lm.evaluate(self.sentences)
        self.assertTrue(net.training)
        net.eval()
        torch.testing.assert_close(net(x), net(x), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            self.lm.save(path)
            restored = m.Model()
            restored.DEVICE = "cpu"
            restored.load(path)
            self.assertEqual(restored.LAYERS, 2)
            self.assertEqual(restored.HEADS, 2)
            self.assertEqual(restored.FFN_HIDDEN, 24)
            self.assertEqual(restored.DROPOUT, 0.5)
            torch.testing.assert_close(net(x), restored.net(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
