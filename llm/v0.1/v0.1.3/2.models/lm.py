# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.3)  -  정규화 · 초기화 (일반화 개선)  (NeuralLM 상속)

[v0.1.2 와 무엇이 다른가]
  모델 구조 · 데이터 파이프라인 · 옵티마이저 '사용법' 은 v0.1.2 그대로예요.
  목표가 달라져요 — 손실을 더 낮추는 게 아니라 **일반화(검증 PPL) 개선** 입니다.
  (v0.0.9 에서 학습 PPL 2.79 vs 검증 PPL 34.39 였던 '외우기 vs 일반화' 격차를 줄이기.)

[무엇을 더하나]
  ① Weight decay (L2 정규화) : 옵티마이저에 weight_decay 한 인자.
       가중치가 너무 커지지 않게 벌점 → 확률 분포가 부드러워짐 → 검증 PPL↓
       (개수 세기 시대의 v0.0.10 스무딩과 같은 정신 — 과신 완화)
  ② 초기화(initialization)   : nn.Linear 시작값을 통제.
       INIT="zeros" 면 0 에서 시작 → 처음엔 '균등 확률'(라플라스 스무딩의 출발점)
       INIT="default" 면 nn.Linear 기본 초기화 그대로.
  ③ (선택) label smoothing   : 정답에 100% 대신 살짝 분산 → 과신을 더 눌러 일반화.
       LABEL_SMOOTHING 으로 켜고 끌 수 있어요 (기본 0.0 = 끔).
  ※ 수치 안정성은 이미 F.cross_entropy(내부 log-sum-exp)가 담당.

[그대로인 것]  BigramModel(nn.Module) · Dataset/DataLoader · torch.optim · model.json 형식 ·
  대화(chat) · 샘플링 · 퍼플렉서티. 그래서 웹앱/평가가 수정 없이 그대로 동작해요.

[주의]  PyTorch 필요.   pip install torch
"""

import os
import importlib.util

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM · BigramModel · build_dataloader 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.1.2" -> "v0.1" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit(
            "이 버전(v0.1.3)은 PyTorch 가 필요해요.\n"
            "  pip install torch\n"
            "(torch 를 설치할 수 있는 파이썬 환경에서 실행해 주세요.)"
        )


# v0.1.2 모듈에서 데이터 파이프라인·NGramLM 계보를 가져옵니다. (모델은 build_net 상속)
_prev = _load_prev_module("v0.1.2")
build_dataloader = _prev.build_dataloader  # (앞,다음) 텐서 -> DataLoader (v0.1.1 것)


# v0.1.2 를 물려받아, '정규화 + 초기화' 를 더해 일반화를 개선합니다.
class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(OPTIMIZER / HIDDEN / LR / EPOCHS / BATCH_SIZE / SEED / WEIGHT_DECAY /
    #  INIT / LABEL_SMOOTHING)는 3.train/train.py 에서 설정해요.

    def make_optimizer(self, model):
        """v0.1.2 의 옵티마이저에 weight_decay(L2 정규화)를 더해 만들어요."""
        params, wd = model.parameters(), self.WEIGHT_DECAY
        if self.OPTIMIZER == "sgd":
            return torch.optim.SGD(params, lr=self.LR, weight_decay=wd)
        if self.OPTIMIZER == "momentum":
            return torch.optim.SGD(params, lr=self.LR, momentum=0.9, weight_decay=wd)
        return torch.optim.Adam(params, lr=self.LR, weight_decay=wd)

    def init_model(self, model):
        """가중치 초기화. INIT='zeros' 면 **마지막 층을 0**으로 → 처음엔 균등 확률에서 출발."""
        if self.INIT == "zeros":
            torch.nn.init.zeros_(model.fc2.weight)
            torch.nn.init.zeros_(model.fc2.bias)

    def train(self, sentences, valid_sentences=None):
        _require_torch()
        # 1) 어휘 + (앞, 다음) 짝
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        xs_list, ys_list = self.make_pairs(sentences)
        xs = torch.tensor(xs_list, dtype=torch.long)
        ys = torch.tensor(ys_list, dtype=torch.long)

        # 2) 데이터로더 (v0.1.1 재사용)
        loader = build_dataloader(xs, ys, self.BATCH_SIZE, shuffle=True)

        # 3) 신경망(2층) + 초기화 + 옵티마이저(weight_decay 포함)
        torch.manual_seed(self.SEED)
        self.net = self.build_net(V, self.HIDDEN)
        self.init_model(self.net)                 # ← 초기화 (v0.1.3)
        optimizer = self.make_optimizer(self.net) # ← weight_decay 포함 (v0.1.3)

        self.losses = []
        stopper = self.start_early_stopping(valid_sentences)
        print(f"  학습 시작: 어휘 {V}개, 은닉 {self.HIDDEN}, 짝 {len(xs_list)}개, epochs {self.EPOCHS}, "
              f"batch {self.BATCH_SIZE}, optim={self.OPTIMIZER}, lr {self.LR}, "
              f"weight_decay {self.WEIGHT_DECAY}, init={self.INIT}, label_smoothing {self.LABEL_SMOOTHING}")

        for epoch in range(1, self.EPOCHS + 1):
            total, n_batches = 0.0, 0
            for batch in loader:
                logits = self.net(batch["input"])         # 순전파 (B, V)
                loss = F.cross_entropy(logits, batch["target"],
                                       label_smoothing=self.LABEL_SMOOTHING)   # ← (v0.1.3)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.item()
                n_batches += 1

            avg = total / n_batches
            self.losses.append(avg)
            if epoch == 1 or epoch % 5 == 0:
                print(f"  epoch {epoch:3d}/{self.EPOCHS}   avg loss {avg:.4f}")
            if self.should_stop_early(stopper, epoch):
                break

        self.finish_early_stopping(stopper)
        self.net.eval()
        return self.net


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")      # 학습용
VALID_PATH = os.path.join(_VERSION_DIR, "1.data", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")                      # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")                    # 어휘
