"""v0.3.0의 데이터·학습·평가·생성을 재사용하고 멀티헤드 엔진만 확장."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[1] / "v0.3.0"


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prev = _module("v031_previous", PREVIOUS / "0.model/lm.py")
_network = _module("v031_network", HERE / "network.py")
MultiHeadLM = _network.MultiHeadLM
sequence_loss = _prev.sequence_loss
_data = _prev._data


class NeuralLM(_prev.NeuralLM):
    MODEL_VERSION = "v0.3.1"
    HEADS = 4

    def build_net(self):
        return MultiHeadLM(len(self.itos), self.EMBED, self.BLOCK_SIZE, self.HEADS)

    def extra_metadata(self):
        # 같은 가중치 모양도 헤드 수에 따라 계산이 달라지므로 반드시 저장합니다.
        return {"heads": self.HEADS}

    def restore_extra_metadata(self, meta):
        self.HEADS = meta["heads"]


NGramLM = NeuralLM
Model = NeuralLM
DATA_PATH, VALID_PATH = _prev.DATA_PATH, _prev.VALID_PATH
MODEL_PATH = str(HERE / "model.pt")
VOCAB_PATH = str(HERE / "vocab.json")
