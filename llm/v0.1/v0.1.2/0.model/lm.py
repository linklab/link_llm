# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.2)  -  torch.optim 옵티마이저로 학습  (NeuralLM 상속)

[v0.1.1 과 무엇이 다른가]
  모델(BigramModel = nn.Module) · 데이터 파이프라인(Dataset/DataLoader) · 확률 엔진은
  **v0.1.1 그대로**예요. 바뀐 건 딱 하나 — **파라미터를 어떻게 갱신하느냐**.
    - v0.1.1: 손으로     for p in model.parameters(): p -= LR * p.grad
    - v0.1.2: PyTorch    optimizer.step()   (torch.optim 이 대신)

[표준 학습 루프 3줄]  (수동 갱신 → 옵티마이저)
    optimizer.zero_grad()   # 이전 기울기 비우기 (예전 p.grad = None 을 대신)
    loss.backward()         # autograd 가 기울기 계산
    optimizer.step()        # 파라미터 갱신 (+ 모멘텀 / 적응 학습률)

[옵티마이저 골라 쓰기]  OPTIMIZER = "sgd" | "momentum" | "adam"
    - SGD          : p -= lr*grad             (v0.1.1 과 같은 규칙, 이제 표준 API 로)
    - SGD+momentum : 이전 방향을 관성처럼 누적 → 지그재그↓, 보통 더 빨리 수렴
    - Adam         : 파라미터마다 학습률 자동 조절 → lr 에 덜 민감, 보통 가장 빠름 (기본값)
  같은 데이터·모델로 셋을 바꿔가며 손실 곡선(self.losses)을 비교해 보세요.
  (SGD/momentum 은 lr 을 훨씬 크게, Adam 은 작게 잡는 게 보통이에요.)

[그대로인 것]  forward · softmax · 대화(chat) · 샘플링 · 퍼플렉서티 · model.json 형식.
  그래서 웹앱/평가가 수정 없이 그대로 동작하고, 결과 모델도 v0.1.0 과 같은 종류예요.

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
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM · BigramModel · build_dataloader 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.1.1" -> "v0.1" (마이너 그룹)
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit(
            "이 버전(v0.1.2)은 PyTorch 가 필요해요.\n"
            "  pip install torch\n"
            "(torch 를 설치할 수 있는 파이썬 환경에서 실행해 주세요.)"
        )


# v0.1.1 모듈에서 데이터 파이프라인·NGramLM 계보를 가져옵니다. (모델은 build_net 상속)
_prev = _load_prev_module("v0.1.1")
build_dataloader = _prev.build_dataloader  # (앞,다음) 텐서 -> DataLoader (v0.1.1 것)


# v0.1.1 을 물려받아, 학습의 '갱신'만 torch.optim 으로 바꿉니다.
class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(OPTIMIZER / HIDDEN / LR / EPOCHS / BATCH_SIZE / SEED)는 1.train/train.py 에서.

    def make_optimizer(self, model):
        """OPTIMIZER 설정에 따라 torch.optim 옵티마이저를 만들어요."""
        params = model.parameters()
        if self.OPTIMIZER == "sgd":
            return torch.optim.SGD(params, lr=self.LR)
        if self.OPTIMIZER == "momentum":
            return torch.optim.SGD(params, lr=self.LR, momentum=0.9)
        return torch.optim.Adam(params, lr=self.LR)          # 기본: Adam

    def train(self, sentences, valid_sentences=None):
        _require_torch()
        # 1) 어휘 + (앞, 다음) 짝  (v0.1.0 의 build_vocab / make_pairs 재사용)
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        xs_list, ys_list = self.make_pairs(sentences)
        xs = torch.tensor(xs_list, dtype=torch.long)
        ys = torch.tensor(ys_list, dtype=torch.long)

        # 2) 데이터로더 (v0.1.1 의 build_dataloader 재사용)
        loader = build_dataloader(xs, ys, self.BATCH_SIZE, shuffle=True)

        # 3) 신경망(2층) + 옵티마이저 (← 이 버전의 핵심)
        torch.manual_seed(self.SEED)              # 초기화 + 셔플 재현 가능하게
        self.net = self.build_net(V, self.HIDDEN)
        optimizer = self.make_optimizer(self.net)

        self.losses = []
        stopper = self.start_early_stopping(valid_sentences)
        print(f"  학습 시작: 어휘 {V}개, 은닉 {self.HIDDEN}, 짝 {len(xs_list)}개, epochs {self.EPOCHS}, "
              f"batch {self.BATCH_SIZE}({len(loader)}개/epoch), optim={self.OPTIMIZER}, lr {self.LR}")

        for epoch in range(1, self.EPOCHS + 1):
            total, n_batches = 0.0, 0
            for batch in loader:                          # DataLoader 가 배치로 묶어 줌
                logits = self.net(batch["input"])         # 순전파 (B, V)
                loss = F.cross_entropy(logits, batch["target"])

                # --- 표준 학습 3줄 (수동 갱신 대신 옵티마이저) ---
                optimizer.zero_grad()                     # 이전 기울기 비우기
                loss.backward()                           # autograd
                optimizer.step()                          # 파라미터 갱신

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


DATA_PATH = os.path.join(_DATA_DIR, "data.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")                      # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")                    # 어휘
