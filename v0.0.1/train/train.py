# -*- coding: utf-8 -*-
"""
train.py  -  나만의 작은 언어 모델(LMM)을 "학습"시키는 프로그램

[핵심 아이디어]
언어 모델이란 "어떤 단어 다음에 어떤 단어가 나올까?"를 맞히는 프로그램이에요.
우리는 아주 간단하게 만들 거예요.

  1. 문장을 단어로 쪼갠다.   예) "나는 밥을 먹습니다" -> ["나는", "밥을", "먹습니다"]
  2. 한 단어 뒤에 어떤 단어가 몇 번 나왔는지 센다.
       "나는" 뒤에 "밥을" 이 몇 번, "학교에" 가 몇 번 ... 이렇게요.
  3. 이 "세어놓은 표"를 파일(model.json)로 저장한다.  <- 이게 바로 '모델'!

즉, 우리의 학습은 그냥 "단어를 세는 것" 이에요. 아주 쉽죠?
"""

import json      # 표(딕셔너리)를 파일로 저장하기 위한 도구
import os        # 폴더/파일 경로를 다루는 도구

# ----------------------------------------------------------------------
# 1) 경로 정하기: 이 파일 기준으로 데이터와 저장 위치를 정합니다.
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # train 폴더 위치
PROJECT = os.path.dirname(HERE)                            # link_lmm 폴더 위치

DATA_PATH = os.path.join(HERE, "data.txt")                # 학습에 쓸 문장들
MODEL_PATH = os.path.join(PROJECT, "models", "model.json")  # 저장할 모델 위치


def read_sentences(path):
    """파일에서 문장들을 한 줄씩 읽어 리스트로 돌려줍니다."""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()      # 앞뒤 공백/줄바꿈 제거
            if line:                 # 빈 줄은 건너뛰기
                sentences.append(line)
    return sentences


def train(sentences):
    """
    문장들을 받아서 "단어 -> 다음 단어 횟수" 표를 만듭니다.

    결과 예시:
    {
      "나는": {"밥을": 1, "학교에": 1, "코딩을": 1, ...},
      "오늘": {"날씨가": 1, "기분이": 1}
    }
    """
    model = {}  # 비어 있는 표에서 시작

    for sentence in sentences:
        words = sentence.split()           # 공백으로 단어 나누기

        # 문장에서 (앞단어, 뒷단어) 짝을 순서대로 살펴봅니다.
        for i in range(len(words) - 1):
            current_word = words[i]        # 지금 단어
            next_word = words[i + 1]       # 바로 다음 단어

            # 지금 단어가 표에 처음 나온 거라면 빈 칸을 만들어 줍니다.
            if current_word not in model:
                model[current_word] = {}

            # 다음 단어의 횟수를 1 늘려줍니다. (없으면 0에서 시작)
            counts = model[current_word]
            counts[next_word] = counts.get(next_word, 0) + 1

    return model


def save_model(model, path):
    """만든 표(모델)를 JSON 파일로 저장합니다."""
    with open(path, "w", encoding="utf-8") as f:
        # ensure_ascii=False -> 한글이 깨지지 않게, indent=2 -> 보기 좋게
        json.dump(model, f, ensure_ascii=False, indent=2)


def main():
    print("데이터를 읽는 중...")
    sentences = read_sentences(DATA_PATH)
    print(f"문장 {len(sentences)}개를 읽었어요.")

    print("학습(단어 세기) 중...")
    model = train(sentences)
    print(f"서로 다른 단어 {len(model)}개를 배웠어요.")

    save_model(model, MODEL_PATH)
    print(f"모델을 저장했어요 -> {MODEL_PATH}")
    print("학습 끝! 이제 test 폴더의 test.py 를 실행해 보세요.")


if __name__ == "__main__":
    main()
