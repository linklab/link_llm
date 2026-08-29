# -*- coding: utf-8 -*-
"""
generate.py  -  '사전학습(Pretraining)'용 산문 코퍼스 생성기 (재현 가능)

실제 LLM 파이프라인을 따라 데이터를 이원화합니다.
  · data/pretrain/ (이 파일) : 산문 — 설명문·이야기·시·상식.  대화 형식/역할 토큰 없음.
                               목적 = 일반 '언어 모델링'(다음 토큰 예측)으로 어휘·문장·지식 습득.
  · data/sft/                : 대화 `<사용자>…<봇>…`.  v0.5(SFT)에서 '어시스턴트 말투'를 입힐 때 사용.

핵심 설계: 산문과 대화가 **같은 어휘/속성**(사과=새콤달콤·비타민, 강아지=멍멍…)을 공유.
→ 나중에 "산문만 알던 모델이 대화를 배우는" SFT 전환이 자연스럽고 극적으로 드러남.

실행:  python3 data/pretrain/generate.py   → data/pretrain/train.txt · data/pretrain/valid.txt + 리포트
"""
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lexicon as G          # 공유 어휘·속성 + count_ppl

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1234
rng = random.Random(SEED)
N_VALID = 400
N_TRAIN = 3600

_lines = []


def emit(s):
    _lines.append(s)


# ── 한국어 조사 헬퍼 (받침 유무로 은/는·이/가·을/를·과/와 선택) ───────────────
def _batchim(word):
    c = word[-1]
    return ("가" <= c <= "힣") and (ord(c) - 0xAC00) % 28 != 0


def J(word, has, no):
    return word + (has if _batchim(word) else no)


def eun(w): return J(w, "은", "는")
def i_ga(w): return J(w, "이", "가")
def eul(w): return J(w, "을", "를")
def gwa(w): return J(w, "과", "와")


def ro(w):
    """방향 조사 로/으로: 받침 없거나 ㄹ받침이면 '로', 그 외 '으로'."""
    c = w[-1]
    if not _batchim(w) or (ord(c) - 0xAC00) % 28 == 8:   # 받침 없음 or ㄹ(종성 8)
        return w + "로"
    return w + "으로"


# ══════════════════════════════════════════════════════════════════════════
# 1) 설명문 — 개체의 '속성'을 산문으로 (대화 데이터와 지식 공유)
# ══════════════════════════════════════════════════════════════════════════
def desc_food():
    for x, (m, n) in G.FOOD.items():
        emit(f"{m} {eun(x)} 많은 사람이 좋아하는 음식이다.")
        emit(f"{eun(x)} {n} 음식이라 자주 먹으면 몸에 좋다.")
        emit(f"{m} {eul(x)} 보면 군침이 돈다. 그래서 나는 {eul(x)} 자주 찾는다.")


def desc_animal():
    for x, (s, c, h) in G.ANIMAL.items():
        emit(f"{c} {eun(x)} '{s}' 하고 운다.")
        emit(f"{eun(x)} {h} 동물이다. 사람들은 {eul(x)} 보며 즐거워한다.")
        emit(f"{c} {x}, {eul(x)} 처음 본 아이는 눈을 반짝였다.")


def desc_place():
    for x, (act, mood) in G.PLACE.items():
        emit(f"{mood} {eun(x)} {act} 하기 좋은 곳이다.")
        emit(f"주말이면 사람들은 {x}에서 {eul(act)} 하며 시간을 보낸다.")


def desc_job():
    for x, d in G.JOB.items():
        emit(f"{eun(x)} {d} 사람이다.")
        emit(f"{d} {eun(x)} 우리 곁에서 꼭 필요한 일을 한다.")


def desc_hobby():
    for x, (feat, eff) in G.HOBBY.items():
        emit(f"{eun(x)} {feat} 활동이다.")
        emit(f"꾸준히 하면 {eun(x)} {eff} 좋은 취미가 된다.")


def desc_plain():
    for x in G.FLOWERS:
        emit(f"{eun(x)} 곱게 피어 향기를 퍼뜨린다.")
    for x in G.TREES:
        emit(f"{eun(x)} 사계절 내내 자리를 지키며 그늘을 드리운다.")
    for x in G.COUNTRIES:
        emit(f"{eun(x)} 먼 곳에 있는 나라다. 저마다 다른 문화가 있다.")
    for x in G.CURIOUS:
        emit(f"{eun(x)} 볼수록 신비롭다. 사람들은 오래전부터 {eul(x)} 궁금해했다.")
    for x in G.NATURE:
        emit(f"{eun(x)} 자연이 만든 풍경이다. 그 앞에 서면 마음이 탁 트인다.")
    for x in G.COLORS:
        emit(f"{eun(x)} 눈을 편안하게 하는 색이다. 나는 {x} 물건을 보면 기분이 좋아진다.")
    for x in G.MUSIC:
        emit(f"{eun(x)} 들으면 마음이 절로 움직인다. 사람들은 오래전부터 {eul(x)} 사랑해 왔다.")
    for x in G.BOOKS:
        emit(f"{eun(x)} 펼치면 새로운 이야기가 시작된다. {eul(x)} 읽는 시간은 늘 짧게 느껴진다.")
    for x in G.SUBJECTS:
        emit(f"{eun(x)} 처음엔 어렵지만 알수록 재미있다. 매일 조금씩 {eul(x)} 익히면 실력이 는다.")
    for x in G.TRANSPORT:
        emit(f"{eun(x)} 사람들을 먼 곳까지 데려다준다. {eun(x)} 우리 생활을 편리하게 한다.")
    for x in G.CLOTHES:
        emit(f"{eun(x)} 날씨와 자리에 맞춰 골라 입는다. 잘 어울리는 {eun(x)} 기분을 밝게 한다.")
    for x in G.TOYS:
        emit(f"{eun(x)} 아이들이 즐겨 가지고 노는 놀잇감이다. {gwa(x)} 함께라면 시간 가는 줄 모른다.")
    for x in G.DRINKS:
        emit(f"{eun(x)} 목을 시원하게 적셔 준다. 더운 날에는 {eun(x)} 더 반갑다.")
    for x in G.DESSERTS:
        emit(f"{eun(x)} 달콤해서 기분을 좋게 한다. 한 입 베어 문 {eun(x)} 하루의 작은 상이다.")
    for x in G.SEASONS:
        emit(f"{eun(x)} 저마다의 빛깔과 냄새를 지닌다. 사람들은 {x} 나름의 즐거움을 찾는다.")


# ══════════════════════════════════════════════════════════════════════════
# 2) 이야기(내러티브) — 여러 개체를 엮은 짧은 산문 (조합으로 대량·다양)
# ══════════════════════════════════════════════════════════════════════════
WEATHER_S = ["하늘이 맑았다", "바람이 선선했다", "햇살이 따뜻했다", "구름이 두둥실 떠 있었다",
             "가랑비가 촉촉이 내렸다", "노을이 붉게 물들었다", "안개가 자욱하게 깔렸다",
             "새들이 지저귀며 아침을 열었다", "먼 산에 옅은 눈이 남아 있었다", "공기가 유난히 맑았다",
             "바람결에 풀 냄새가 실려 왔다", "빗방울이 창을 두드렸다", "햇빛이 눈부시게 쏟아졌다",
             "저녁놀이 하늘을 물들였다"]
FEEL_S = ["참 즐거운 하루였다", "오래 기억에 남을 것 같다", "마음이 포근해졌다",
          "괜히 콧노래가 나왔다", "내일이 벌써 기다려졌다", "작은 행복이 밀려왔다",
          "하루가 선물처럼 느껴졌다", "가슴이 뭉클해졌다", "피곤도 어느새 사라졌다",
          "문득 고마운 마음이 들었다", "오늘도 한 뼘 자란 기분이었다", "잔잔한 여운이 남았다",
          "저절로 미소가 지어졌다", "돌아오는 길이 가벼웠다"]
# 내러티브 문장 위치별 '틀' 은행 — 틀 수를 크게 늘려 특정 바이그램이 지배하지 않게 함
GO_S = ["아침 일찍 {p}에 갔다", "오늘은 {p}에서 하루를 보냈다", "친구와 {p}에 들렀다",
        "오랜만에 {eul} 찾았다", "발길이 자연스레 {ro} 향했다", "점심을 먹고 {p}에 갔다",
        "느지막이 일어나 {ro} 나섰다", "가족과 함께 {eul} 거닐었다"]
MEET_S = ["{ga} 반갑게 다가왔다", "{eul} 만나 한참을 바라보았다", "멀리서 {ga} 뛰어노는 모습이 보였다",
          "{eul} 보자 마음이 환해졌다", "{ga} 곁으로 살금살금 다가왔다", "{eul} 오래도록 지켜보았다",
          "{ga} 고개를 갸웃하며 나를 살폈다", "{eul} 조심스레 쓰다듬어 보았다"]
EAT_S = ["점심으로 {eul} 맛있게 먹었다", "출출해서 {eul} 조금 나눠 먹었다", "{eun} 생각보다 훨씬 맛있었다",
         "간식으로 {eul} 챙겨 먹었다", "{eul} 한 입 베어 무니 기분이 좋아졌다",
         "따뜻한 {eul} 천천히 음미했다", "{eun} 오늘따라 유난히 달았다"]
DO_S = ["오후에는 {eul} 하며 시간을 보냈다", "틈틈이 {eul} 즐겼다", "해가 질 때까지 {eul} 이어 갔다",
        "새로 {eul} 배워 보기로 했다", "{eul} 하다 보니 시간 가는 줄 몰랐다"]


def _fill(frame, **kw):
    return frame.format(**kw)


def narrative():
    place = rng.choice(list(G.PLACE))
    animal = rng.choice(list(G.ANIMAL))
    food = rng.choice(list(G.FOOD))
    hobby = rng.choice(list(G.HOBBY))
    # 첫 문장은 '개체(장소)가 든 문장'으로 → 시작이 다양해짐(날씨 문장 반복 제거).
    s = [_fill(rng.choice(GO_S), p=place, eul=eul(place), ro=ro(place)) + "."]
    # 중간 절들은 만들어 두고 '순서를 섞어' 넣음 → 같은 얼개 반복을 줄임.
    mids = [
        _fill(rng.choice(MEET_S), ga=i_ga(animal), eul=eul(animal)) + ".",
        _fill(rng.choice(EAT_S), eul=eul(food), eun=eun(food)) + ".",
    ]
    if rng.random() < 0.5:
        mids.append(_fill(rng.choice(DO_S), eul=eul(hobby)) + ".")
    if rng.random() < 0.45:
        mids.append(rng.choice(WEATHER_S) + ".")   # 날씨는 '중간 관찰'로
    rng.shuffle(mids)
    s += mids
    s.append(rng.choice(FEEL_S) + ".")
    return " ".join(s)


# ══════════════════════════════════════════════════════════════════════════
# 3) 짧은 시 / 동요 — 운율 있는 산문
# ══════════════════════════════════════════════════════════════════════════
def poem():
    flower = rng.choice(G.FLOWERS)
    animal = rng.choice(list(G.ANIMAL))
    season = rng.choice(["봄", "여름", "가을", "겨울"])
    color = rng.choice(G.COLORS)
    forms = [
        f"{season} 바람 살랑살랑 / {flower} 방긋 피었네 / {i_ga(animal)} 폴짝 뛰노네",
        f"{color} 하늘 아래 / {flower} 한 송이 / 오늘도 곱게 웃네",
        f"들판에 {flower} / 숲속엔 {animal} / {season}이 성큼 왔네",
        f"반짝반짝 별빛 아래 / {i_ga(animal)} 잠이 들고 / {flower}도 꿈을 꾸네",
    ]
    return rng.choice(forms)


# ══════════════════════════════════════════════════════════════════════════
# 4) 상식 · 짧은 수필 (직접 작성 — 일반 지식/문장 감각)
# ══════════════════════════════════════════════════════════════════════════
FACTS = [
    "해는 동쪽에서 떠서 서쪽으로 진다.", "물은 높은 곳에서 낮은 곳으로 흐른다.",
    "달은 밤하늘에서 환하게 빛난다.", "봄이 지나면 여름이 온다.",
    "비가 온 뒤에는 무지개가 뜨기도 한다.", "나무는 햇빛을 받아 무럭무럭 자란다.",
    "새는 하늘을 자유롭게 날아다닌다.", "겨울에는 눈이 내리고 날씨가 춥다.",
    "바다는 넓고 푸르며 짠맛이 난다.", "별은 밤이 되면 하나둘 모습을 드러낸다.",
    "꽃은 봄이 되면 앞다투어 피어난다.", "개미는 부지런히 먹이를 나른다.",
    "낮에는 해가, 밤에는 달이 하늘을 지킨다.", "씨앗을 심고 물을 주면 싹이 튼다.",
    "가을이 오면 나뭇잎이 붉고 노랗게 물든다.", "얼음은 따뜻해지면 물로 녹는다.",
    "무지개는 일곱 빛깔로 이루어져 있다.", "바람이 불면 나뭇가지가 흔들린다.",
    "벌은 꽃에서 꿀을 모아 벌집으로 돌아간다.", "구름이 모이면 비가 내린다.",
]
ESSAYS = [
    "나는 조용한 아침을 좋아한다. 창밖에서 새소리가 들리면 마음이 편안해진다.",
    "비 오는 날에는 따뜻한 차 한 잔이 생각난다. 빗소리를 들으면 하루가 느리게 흐른다.",
    "작은 습관 하나가 하루를 바꾼다. 아침에 물 한 잔을 마시는 것부터 시작해 본다.",
    "책을 읽으면 가 보지 못한 세상을 만난다. 한 장 한 장이 새로운 길이 된다.",
    "실수는 부끄러운 것이 아니다. 실수에서 배우면 어제보다 한 뼘 자란다.",
    "느리게 걸어도 괜찮다. 방향만 맞다면 언젠가는 닿는다.",
    "친구와 나눈 짧은 인사가 온종일 힘이 될 때가 있다.",
    "정리를 하고 나면 마음까지 정돈되는 기분이 든다.",
    "좋아하는 일을 꾸준히 하면 그 일이 나를 닮아 간다.",
    "밤이 깊을수록 별은 더 또렷하게 빛난다.",
]


def build_all():
    _lines.clear()
    desc_food(); desc_animal(); desc_place(); desc_job(); desc_hobby(); desc_plain()
    for _ in range(4000):
        emit(narrative())
    for _ in range(1500):
        emit(poem())
    _lines.extend(FACTS)
    _lines.extend(ESSAYS)
    seen, uniq = set(), []
    for ln in _lines:
        if ln not in seen:
            seen.add(ln)
            uniq.append(ln)
    return uniq


def main():
    pool = build_all()
    rng.shuffle(pool)
    need = N_TRAIN + N_VALID
    if len(pool) < need:
        raise SystemExit(f"생성 풀 부족: {len(pool)} < {need}. 이야기/시 반복수를 늘리세요.")
    # 정렬하지 않음(정렬하면 같은 첫 문장끼리 뭉쳐 보임). 이미 셔플된 pool 순서를 그대로 사용.
    valid = pool[:N_VALID]
    train = pool[N_VALID:N_VALID + N_TRAIN]
    with open(os.path.join(HERE, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(os.path.join(HERE, "valid.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(valid) + "\n")

    vocab = set(" ".join(train + valid).split())
    lens = [len(ln.split()) for ln in train]
    cppl = G.count_ppl(train, valid)
    print(f"[사전학습 산문] 풀 {len(pool)} → 학습 {len(train)} / 검증 {len(valid)} (겹침 {len(set(train)&set(valid))})")
    print(f"  어휘(토큰 종류): {len(vocab)}")
    print(f"  평균 어절: {sum(lens)/len(lens):.1f}  (min {min(lens)} / max {max(lens)})")
    print(f"  난이도 게이지(카운트 검증 PPL 근사): {cppl:.1f}")


if __name__ == "__main__":
    main()
