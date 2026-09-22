"""v0.3.5 캡스톤: 추론 모델 구조는 v0.3.4를 유지합니다."""
import importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v035_previous", HERE.parents[1] / "v0.3.4/0.model/lm.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
class NeuralLM(previous.NeuralLM):
    MODEL_VERSION = "v0.3.5"
NGramLM = Model = NeuralLM
DATA_PATH, VALID_PATH = previous.DATA_PATH, previous.VALID_PATH
MODEL_PATH = str(HERE / "model.pt")
VOCAB_PATH = str(HERE / "vocab.json")
