"""v0.3.4: 다층 decoder-only 미니 GPT. 이전 버전의 공통 학습·평가를 재사용합니다."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / "v0.3.3"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prev = _module("v034_previous", PREVIOUS / "0.model/lm.py")
_network = _module("v034_network", HERE / "network.py")
TransformerLM = _network.TransformerLM
sequence_loss = _prev.sequence_loss
_data = _prev._data


class NeuralLM(_prev.NeuralLM):
    MODEL_VERSION = "v0.3.4"
    LAYERS = 4
    HEADS = 4
    FFN_HIDDEN = 256
    DROPOUT = 0.1

    def build_net(self):
        return TransformerLM(len(self.itos), self.EMBED, self.BLOCK_SIZE, self.HEADS,
                             self.FFN_HIDDEN, self.DROPOUT, self.LAYERS)

    def extra_metadata(self):
        # 같은 가중치 모양도 헤드 수에 따라 계산이 달라지므로 반드시 저장합니다.
        return {"layers": self.LAYERS, "heads": self.HEADS, "ffn_hidden": self.FFN_HIDDEN, "dropout": self.DROPOUT}

    def restore_extra_metadata(self, meta):
        self.LAYERS = meta["layers"]
        self.HEADS = meta["heads"]
        self.FFN_HIDDEN = meta["ffn_hidden"]
        self.DROPOUT = meta["dropout"]


NGramLM = NeuralLM
Model = NeuralLM
DATA_PATH, VALID_PATH = _prev.DATA_PATH, _prev.VALID_PATH
MODEL_PATH = str(HERE / "model.pt")
VOCAB_PATH = str(HERE / "vocab.json")
