# -*- coding: utf-8 -*-
"""
lm.py  (v0.1.1)  -  PyTorch Dataset + DataLoader 로 미니배치 학습  (NeuralLM 상속)

[v0.1.0 과 무엇이 다른가]
  모델(신경망 bigram, 가중치 W)과 확률 엔진은 **v0.1.0 그대로**예요.
  바뀐 건 딱 하나 — **어떻게 학습 데이터를 공급하느냐(train)** 입니다.
    - v0.1.0: 전체 데이터를 한 번에(full-batch) 써서 경사하강.
    - v0.1.1: **PyTorch 의 `Dataset` + `DataLoader`** 로 데이터를 감싸,
              DataLoader 가 **섞기(shuffle) + 미니배치(mini-batch)** 를 대신 해줘요.

[강의 자료의 관례를 그대로 따릅니다]  (03.real_world_data_to_tensors)
  1) `Dataset` 을 상속해 세 가지만 정의:
       __init__     : data / target 을 준비(읽기·전처리)
       __len__      : 샘플 개수
       __getitem__  : 한 샘플을 **딕셔너리 {'input': x, 'target': y}** 로 반환
  2) `DataLoader(dataset=..., batch_size=..., shuffle=True)` 로 감싸고,
  3) `for batch in loader:` 로 돌면 `batch['input']`, `batch['target']` 가 배치로 나와요.
  → "실제 데이터를 텐서로 바꿔 모델에 흘려보내는" 표준 방식이라, 앞으로(v0.1.x~) 계속 씁니다.

[그대로인 것]  forward · softmax · 대화(chat) · 샘플링 · 퍼플렉서티 · model.json 형식은
  v0.1.0/v0.0.x 에서 물려받아요. 그래서 학습 파이프라인만 바뀌고 **결과 모델은 v0.1.0 과 같은 종류**.

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
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 과 BigramModel 을 함께 재사용하려고)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.1.0" -> "v0.1" (마이너 그룹)
    lmm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../lmm 루트
    path = os.path.join(lmm_dir, group, prev_version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("lmmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit(
            "이 버전(v0.1.1)은 PyTorch 가 필요해요.\n"
            "  pip install torch\n"
            "(torch 를 설치할 수 있는 파이썬 환경에서 실행해 주세요.)"
        )


# ---------- 데이터 파이프라인은 1.data/dataloader.py 에 분리해 두고 불러옵니다 ----------
def _load_data_module():
    """1.data/dataloader.py (BigramDataset + build_dataloader) 를 경로로 불러와요."""
    path = os.path.join(_VERSION_DIR, "1.data", "dataloader.py")
    spec = importlib.util.spec_from_file_location("lmm_v0_1_1_dataloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_data = _load_data_module()
BigramDataset = _data.BigramDataset          # (참고용) 데이터셋 클래스
build_dataloader = _data.build_dataloader    # (앞,다음) 텐서 -> DataLoader

# v0.1.0 모듈에서 신경망 모델(BigramModel = nn.Module)과 NGramLM 계보를 함께 가져옵니다.
_prev = _load_prev_module("v0.1.0")
BigramModel = _prev.BigramModel


# v0.1.0(신경망 bigram)을 물려받아, 학습을 Dataset + DataLoader 방식으로 바꿉니다.
class NeuralLM(_prev.NGramLM):
    # ----- 하이퍼파라미터 (필요하면 조정) -----
    LR = 10.0          # 학습률
    EPOCHS = 300       # 전체 데이터를 몇 번 반복(에폭)
    BATCH_SIZE = 64    # 한 배치에 담는 (앞,다음) 짝의 수
    SEED = 1234        # 초기 W + DataLoader 셔플을 재현 가능하게

    def train(self, sentences):
        _require_torch()
        # 1) 어휘 사전 + (앞, 다음) 짝  (v0.1.0 의 build_vocab / make_pairs 재사용)
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        xs_list, ys_list = self.make_pairs(sentences)
        xs = torch.tensor(xs_list, dtype=torch.long)
        ys = torch.tensor(ys_list, dtype=torch.long)

        # 2) 데이터로더 (1.data/dataloader.py 가 Dataset+DataLoader 를 한 번에 만들어 줌)
        loader = build_dataloader(xs, ys, self.BATCH_SIZE, shuffle=True)

        # 3) 모델 (nn.Module) — v0.1.0 의 BigramModel 을 그대로 재사용.
        torch.manual_seed(self.SEED)              # 초기화 + 셔플 재현 가능하게
        model = BigramModel(V)

        self.losses = []   # 에폭별 평균 손실 (손실 곡선 확인용)
        print(f"  학습 시작: 어휘 {V}개, 짝 {len(xs_list)}개, epochs {self.EPOCHS}, "
              f"batch {self.BATCH_SIZE}({len(loader)}개/epoch), lr {self.LR}")

        for epoch in range(1, self.EPOCHS + 1):
            total, n_batches = 0.0, 0
            for batch in loader:                          # DataLoader 가 배치로 묶어 줌
                bx = batch["input"]                       # (B,)  앞 토큰
                by = batch["target"]                      # (B,)  정답 토큰
                # BATCH_SIZE가 4일 때
                # tensor([  2,   0,   4, 162])
                # tensor([258, 512,   5, 112])

                # --- 순전파: model(bx) 가 forward() 를 불러 logits 를 냄 ---
                logits = model(bx)                        # (B, V)
                loss = F.cross_entropy(logits, by)

                # --- 역전파 + 경사하강 1스텝 (배치마다) ---
                for p in model.parameters():
                    p.grad = None
                loss.backward()
                with torch.no_grad():
                    for p in model.parameters():
                        p -= self.LR * p.grad

                total += loss.item()
                n_batches += 1

            avg = total / n_batches
            self.losses.append(avg)
            if epoch == 1 or epoch % 5 == 0:
                print(f"  epoch {epoch:3d}/{self.EPOCHS}   avg loss {avg:.4f}")

        # self.W[prev] = 그 앞 토큰 다음의 logits (nn.Linear.weight 를 전치해 맞춤)
        self.W = model.linear.weight.detach().t().contiguous()
        return self.W


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


DATA_PATH = os.path.join(_VERSION_DIR, "1.data", "data.txt")      # 학습용
VALID_PATH = os.path.join(_VERSION_DIR, "1.data", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.json")
