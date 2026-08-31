# -*- coding: utf-8 -*-
"""
test.py  (v0.2.4)  -  임베딩 시각화 · 분석

이 버전의 평가는 PPL 이 아니라 **임베딩을 들여다보는 것**이에요 (모델은 v0.2.3 그대로).

  ① 최근접이웃 — 코사인 유사도로 "이 토큰과 가까운 토큰들"
  ② PCA        — 2차원 산점도 → 2.test/embedding_pca.svg
  ③ t-SNE      — 2차원 산점도 → 2.test/embedding_tsne.svg

※ PyTorch 필요:  pip install torch
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "llmlm_" + os.path.basename(_HERE).replace(".", "_"),
    os.path.join(_HERE, "0.model", "lm.py"),
)
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
Model = _model.Model
DATA_PATH = _model.DATA_PATH
VALID_PATH = _model.VALID_PATH
MODEL_PATH = _model.MODEL_PATH
OUT_DIR = os.path.join(_HERE, "2.test")

# 최근접이웃을 살펴볼 씨앗 토큰 (어휘에 없으면 자동으로 건너뜁니다)
# 조사가 분리된 뒤로는 **어간**이 그대로 토큰이에요 (v0.0.6 토크나이저).
PROBES = ["강아지", "먹었다", "따뜻한", "바다", "사과", "기분"]


def main():
    lm = Model.load_or_exit(MODEL_PATH)
    train_sents = lm.read_sentences(DATA_PATH)
    E = lm.embedding_matrix()

    print("=== v0.2.4 (임베딩 시각화) ===\n")
    print(f"임베딩 표 : {tuple(E.shape)}  (어휘 V × 임베딩 차원 E)")
    print(f"검증 PPL  : {lm.perplexity(lm.read_sentences(VALID_PATH)):.2f}"
          f"   ← 모델을 안 바꿨으니 v0.2.3 과 같아야 정상\n")

    # ① 최근접이웃 — 임베딩이 의미를 담았는지 가장 직접적으로 보여줘요
    print("--- ① 최근접이웃 (코사인 유사도, 1에 가까울수록 비슷) ---")
    found = False
    for tok in PROBES:
        near = lm.nearest(tok, k=5)
        if near is None:
            print(f"  [{tok}] 어휘에 없어요 — 건너뜀")
            continue
        found = True
        pretty = " · ".join(f"{t}({v:.2f})" for t, v in near)
        print(f"  [{tok}] → {pretty}")
    if found:
        print("  → 같은 부류(음식·동물·장소…)끼리 모이면 '임베딩이 의미를 배웠다'는 신호예요.\n")

    # 산점도에 찍을 토큰: 학습 데이터에서 자주 나온 순
    idx = lm.plot_tokens(train_sents)
    labels = [lm.itos[i] for i in idx]

    # ② PCA — 선형, 빠름, '분산을 얼마나 남겼나'를 숫자로 알려줌
    print("--- ② PCA (선형 투영) ---")
    coords, ratio = lm.pca_2d(idx)
    path = lm.save_embedding_plot(
        os.path.join(OUT_DIR, "embedding_pca.svg"), coords, labels,
        title=f"v0.2.4 임베딩 PCA (상위 {len(idx)}개 토큰)",
        note=f"설명된 분산 {ratio * 100:.1f}%")
    print(f"  설명된 분산 : {ratio * 100:.1f}%   ← 2차원이 원본 정보의 몇 %를 담았나")
    print(f"  저장        : {path}\n")

    # ③ t-SNE — 비선형, 이웃 관계 보존, autograd 로 KL 최소화
    print("--- ③ t-SNE (비선형, 이웃 보존) ---")
    coords, kl = lm.tsne_2d(idx, log=print)
    path = lm.save_embedding_plot(
        os.path.join(OUT_DIR, "embedding_tsne.svg"), coords, labels,
        title=f"v0.2.4 임베딩 t-SNE (상위 {len(idx)}개 토큰)",
        note=f"perplexity {lm.TSNE_PERPLEXITY:g} · KL {kl:.3f}")
    print(f"  최종 KL     : {kl:.4f}   ← 낮을수록 이웃 관계를 잘 보존")
    print(f"  저장        : {path}\n")

    print("→ 두 그림을 나란히 보세요. PCA 는 전체 퍼짐을, t-SNE 는 **덩어리(군집)** 를 잘 보여줘요.")
    print("  단, t-SNE 의 덩어리 사이 '거리'는 의미가 없어요 — 이웃 관계만 믿을 것.")
    print("  (SVG 는 벡터라 브라우저에서 확대해도 글자가 깨지지 않아요.)")


if __name__ == "__main__":
    main()
