# -*- coding: utf-8 -*-
"""
lm.py  (v0.2.3)  -  일반화 · 튜닝 (dropout · weight tying · LR 스케줄)

[이 버전이 더하는 것 — 도구 3개]
  ① **dropout**      : 학습 중 은닉 뉴런 일부를 무작위로 꺼서 특정 경로에 의존하지 못하게.
                       평가/생성 때는 자동으로 꺼져요(`net.eval()`).
  ② **weight tying** : 출력층 `fc2.weight` 를 입력 임베딩 `emb.weight` 와 **같은 텐서로 공유**.
                       파라미터가 V×H 만큼 줄고, "같은 토큰은 입력에서도 출력에서도 같은 벡터"가 됩니다.
                       ※ 모양이 맞아야 하므로 **HIDDEN == EMBED** 가 필수예요.
  ③ **LR 스케줄**    : 학습이 진행될수록 학습률을 줄여 마지막에 곱게 수렴시켜요.
                       "none" | "cosine" | "step" | "plateau" 중 선택.

[구조]
  (앞 N토큰) → 임베딩 concat(N·E) → fc1 → [BatchNorm(선택)] → tanh → [Dropout] → fc2 → logits(V)
                                                                                  ↑ (선택) emb.weight 공유

[그대로인 것]  임베딩·문맥(block_size)·데이터·옵티마이저·조기 종료·대화·PPL 은 v0.2.2 상속.
               바꾼 건 build_net·train(스케줄러)·load 뿐.

[주의]  PyTorch 필요.   pip install torch
"""

import os
import json
import importlib.util

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    torch = None
    F = None
    nn = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_VERSION_DIR = os.path.dirname(_HERE)


def _load_prev_module(prev_version):
    """이전 버전 lm.py 모듈을 통째로 불러와요 (NGramLM 계보 재사용)."""
    group = prev_version.rsplit(".", 1)[0]                    # "v0.2.2" -> "v0.2"
    llm_dir = os.path.dirname(os.path.dirname(_VERSION_DIR))  # .../llm 루트
    path = os.path.join(llm_dir, group, prev_version, "0.model", "lm.py")
    spec = importlib.util.spec_from_file_location("llmlm_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_data_module():
    """공용 data/dataloader.py (BigramDataset + build_dataloader) 를 경로로 불러와요. (v0.1.1 과 같은 방식)"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))   # 저장소 루트
    path = os.path.join(root, "data", "dataloader.py")
    spec = importlib.util.spec_from_file_location("llm_v0_2_3_dataloader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_torch():
    if torch is None:
        raise SystemExit("이 버전(v0.2.3)은 PyTorch 가 필요해요.\n  pip install torch")


# v0.2.2(초기화·정규화)를 물려받아, 일반화·튜닝 도구만 더합니다.
_prev = _load_prev_module("v0.2.2")
build_dataloader = _load_data_module().build_dataloader    # (앞,다음) 텐서 -> DataLoader (공용)

_Module = nn.Module if torch is not None else object   # torch 없이도 이 파일이 로딩되게


# ---------- N토큰 임베딩 MLP + BatchNorm(선택) + Dropout + weight tying(선택) ----------
class TuneModel(_Module):
    """
    v0.2.2 의 NormModel 에 **Dropout** 과 **weight tying** 을 더한 모델.

      forward: emb(x).flatten(1) → fc1 → [BN] → tanh → [Dropout] → fc2 → logits

    - `dropout=0.0` 이면 Dropout 층이 항등(identity)이라 v0.2.2 와 완전히 같아요.
    - `tie_weights=True` 면 `fc2.weight` 가 `emb.weight` 를 **가리킵니다**(복사가 아니라 공유).
      두 층이 한 텐서를 함께 학습하므로 파라미터가 V×H 만큼 줄어요.
    """
    def __init__(self, vocab_size, hidden, embed, block_size,
                 use_bn=False, dropout=0.0, tie_weights=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, embed)
        self.fc1 = nn.Linear(block_size * embed, hidden, bias=not use_bn)
        self.bn = nn.BatchNorm1d(hidden) if use_bn else None
        self.drop = nn.Dropout(dropout)                # p=0.0 이면 아무 일도 안 해요
        self.fc2 = nn.Linear(hidden, vocab_size)
        self.tied = bool(tie_weights)
        if self.tied:
            if hidden != embed:
                raise SystemExit(
                    "weight tying 은 출력층(V×HIDDEN)과 임베딩(V×EMBED)의 모양이 같아야 해요.\n"
                    f"  지금은 HIDDEN={hidden}, EMBED={embed} 로 다릅니다.\n"
                    "  1.train/train.py 에서 HIDDEN 과 EMBED 를 같게 맞추거나 TIE_WEIGHTS=False 로 두세요."
                )
            self.fc2.weight = self.emb.weight          # ★ 같은 텐서를 공유 (복사 아님)

    def forward(self, x):                              # x: (B, N)
        z = self.fc1(self.emb(x).flatten(1))           # (B, hidden)
        if self.bn is not None:
            z = self.bn(z)
        return self.fc2(self.drop(torch.tanh(z)))      # logits (B, V)


class NeuralLM(_prev.NGramLM):
    # 하이퍼파라미터(DROPOUT / TIE_WEIGHTS / LR_SCHEDULE / ... )는 1.train/train.py 에서 설정해요.
    DROPOUT = 0.0             # 은닉층 뒤 dropout 확률 (0.0 = 끔)
    TIE_WEIGHTS = False       # 출력층과 임베딩 가중치 공유 (HIDDEN == EMBED 필요)
    LR_SCHEDULE = "none"      # "none" | "cosine" | "step" | "plateau"
    LR_MIN = 0.0              # cosine 이 내려갈 바닥 학습률
    LR_STEP = 50              # step: 몇 에폭마다 줄일지
    LR_GAMMA = 0.5            # step: 줄이는 비율
    LR_PATIENCE = 5           # plateau: 몇 에폭 정체하면 줄일지

    def build_net(self, vocab_size, hidden):
        return TuneModel(vocab_size, hidden, self.EMBED, self.BLOCK_SIZE,
                         use_bn=self.USE_BN, dropout=self.DROPOUT,
                         tie_weights=self.TIE_WEIGHTS)

    # ---------- LR 스케줄 ----------
    def make_scheduler(self, optimizer):
        """
        학습률을 에폭마다 줄이는 스케줄러를 만들어요. (LR_SCHEDULE="none" 이면 None)

          cosine  : 코사인 곡선으로 LR → LR_MIN 까지 부드럽게 감소 (가장 무난)
          step    : LR_STEP 에폭마다 LR_GAMMA 배로 계단식 감소
          plateau : **검증 PPL 이 좋아지지 않을 때만** 감소 (조기 종료와 같은 신호를 봄)
        """
        s = self.LR_SCHEDULE
        if s == "none":
            return None
        if s == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.EPOCHS, eta_min=self.LR_MIN)
        if s == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.LR_STEP, gamma=self.LR_GAMMA)
        if s == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.LR_GAMMA, patience=self.LR_PATIENCE)
        raise SystemExit(f'LR_SCHEDULE 는 "none"/"cosine"/"step"/"plateau" 중 하나여야 해요 (지금: {s})')

    def step_scheduler(self, scheduler):
        """에폭 끝에서 스케줄러를 한 칸 진행해요.
        plateau 만 '무엇을 볼지'가 필요해서 검증 PPL 을 넘겨줍니다."""
        if scheduler is None:
            return
        if self.LR_SCHEDULE == "plateau":
            if self.valid_scores:                      # should_stop_early 가 방금 채워 둔 값
                scheduler.step(self.valid_scores[-1])
        else:
            scheduler.step()

    # ---------- 학습 (v0.1.3 루프 + 스케줄러) ----------
    def train(self, sentences, valid_sentences=None):
        _require_torch()
        # 1) 어휘 + (앞 N토큰, 다음) 짝
        self.itos = self.build_vocab(sentences)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        xs_list, ys_list = self.make_pairs(sentences)
        xs = torch.tensor(xs_list, dtype=torch.long, device=self.device())
        ys = torch.tensor(ys_list, dtype=torch.long, device=self.device())

        # 2) 데이터로더 (v0.1.1 재사용)
        loader = build_dataloader(xs, ys, self.BATCH_SIZE, shuffle=True)

        # 3) 신경망 + 초기화 + 옵티마이저 + 스케줄러
        torch.manual_seed(self.SEED)
        self.net = self.make_net(V, self.HIDDEN)
        self.init_model(self.net)
        optimizer = self.make_optimizer(self.net)
        scheduler = self.make_scheduler(optimizer)     # ← 새로 추가 (v0.2.3)

        self.losses = []
        self.lrs = []                                  # 에폭별 학습률 (스케줄이 보이도록)
        stopper = self.start_early_stopping(valid_sentences)
        params = sum(p.numel() for p in self.net.parameters())
        print(f"  학습 시작: 어휘 {V}개, 은닉 {self.HIDDEN}, 임베딩 {self.EMBED}, 문맥 {self.BLOCK_SIZE}, "
              f"짝 {len(xs_list)}개, epochs {self.EPOCHS}, batch {self.BATCH_SIZE}")
        print(f"  튜닝: dropout {self.DROPOUT}, weight_tying {self.TIE_WEIGHTS}, "
              f"lr_schedule={self.LR_SCHEDULE}, lr {self.LR}, weight_decay {self.WEIGHT_DECAY}, "
              f"파라미터 {params:,}개")

        for epoch in range(1, self.EPOCHS + 1):
            self.net.train()                           # ← dropout 켜기 (학습 모드)
            total, n_batches = 0.0, 0
            for batch in loader:
                logits = self.net(batch["input"])
                loss = F.cross_entropy(logits, batch["target"],
                                       label_smoothing=self.LABEL_SMOOTHING)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                n_batches += 1

            avg = total / n_batches
            self.losses.append(avg)
            self.lrs.append(optimizer.param_groups[0]["lr"])
            if epoch == 1 or epoch % 5 == 0:
                print(f"  epoch {epoch:3d}/{self.EPOCHS}   avg loss {avg:.4f}   "
                      f"lr {optimizer.param_groups[0]['lr']:.6f}")
            stop = self.should_stop_early(stopper, epoch)   # 검증 PPL 을 여기서 재요
            self.step_scheduler(scheduler)                  # ← 그 값을 plateau 가 씁니다
            if stop:
                break

        self.finish_early_stopping(stopper)
        self.net.eval()                                # ← dropout 끄기 (평가/생성 모드)
        return self.net

    # ---------- 저장 / 불러오기 ----------
    def save(self, model_path, vocab_path):
        """
        가중치를 저장해요. **weight tying 을 켰으면 `fc2.weight` 를 빼고 저장**합니다.

        `fc2.weight` 는 `emb.weight` 와 **같은 텐서**라, 그대로 두면 파일에 같은 값이 두 번 들어가요
        (V×H = 15.7만 개가 헛되이). 빼고 저장하면 파일도 파라미터 수만큼 실제로 줄어듭니다.
        불러올 때는 `emb.weight` 를 읽어 넣으면 공유된 `fc2.weight` 도 함께 채워져요.
        """
        _require_torch()
        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        if getattr(self.net, "tied", False):
            state.pop("fc2.weight", None)          # 공유 텐서는 한 번만 저장
        torch.save(state, model_path)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({"tokenizer": self.tokenizer_name(), "vocab": self.itos},
                      f, ensure_ascii=False)

    def load(self, model_path):
        """
        model.pt + vocab.json 로 복원.

        저장된 가중치 모양에서 E·N·H·BatchNorm 유무를 되살려요. **weight tying 여부**는
        `fc2.weight` 가 파일에 **없으면** 켜진 것으로 판별합니다(save 가 공유 텐서를 뺐으므로).
        (dropout 은 평가 때 어차피 꺼지므로 복원할 필요가 없어요.)
        """
        _require_torch()
        vocab_path = os.path.join(os.path.dirname(model_path), "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.itos = meta["vocab"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        V = len(self.itos)

        state = torch.load(model_path, map_location="cpu")
        self.EMBED = state["emb.weight"].shape[1]
        self.BLOCK_SIZE = state["fc1.weight"].shape[1] // self.EMBED
        self.USE_BN = "bn.weight" in state
        hidden = state["fc1.weight"].shape[0]
        # tied 판별: save 가 공유 텐서를 뺐으므로 `fc2.weight` 가 **없으면** tied.
        # (모양이 같고 값도 같은 파일도 tied 로 봐요 — 공유 텐서를 두 번 저장한 형식까지 읽히게.)
        self.TIE_WEIGHTS = ("fc2.weight" not in state) or (
            hidden == self.EMBED and torch.equal(state["emb.weight"], state["fc2.weight"]))
        self.DROPOUT = 0.0                             # 평가 모드에선 무의미
        self.net = self.make_net(V, hidden)
        # strict=False : tied 면 fc2.weight 가 파일에 없어요. emb.weight 를 채우면
        #                같은 텐서인 fc2.weight 도 함께 채워지므로 문제 없습니다.
        self.net.load_state_dict(state, strict=self.TIE_WEIGHTS is False)
        self.net.eval()
        return self


# web_service / 상속 사슬이 module.NGramLM 을 찾으므로 노출.
NGramLM = NeuralLM


class Model(NeuralLM):
    pass


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터
DATA_PATH = os.path.join(_DATA_DIR, "pretrain", "train.txt")      # 학습용
VALID_PATH = os.path.join(_DATA_DIR, "pretrain", "valid.txt")    # 검증용
MODEL_PATH = os.path.join(_HERE, "model.pt")         # 가중치 (PyTorch 표준)
VOCAB_PATH = os.path.join(_HERE, "vocab.json")       # 어휘 (0번=<PAD>)
