# -*- coding: utf-8 -*-
"""
lm.py  (v0.0.1)  -  기반 클래스 NGramLM + 이 버전 설정 (base.py 와 model.py 를 하나로 합침)

- NGramLM : 언어 모델의 모든 기능 (문장 읽기·학습·저장·불러오기·생성). 뒤 버전들이 물려받아요.
- Model   : 이 버전의 설정(ORDERS)을 정한, 실제로 쓰는 클래스.

[v0.0.1 의 성격]  앞 1단어 / 그리디 / 문장끝 없음 / 백오프 없음 / 띄어쓰기
"""

import json
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))     # 이 버전의 models 폴더
_VERSION_DIR = os.path.dirname(_HERE)                  # 이 버전 폴더 (v0.0.1)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_VERSION_DIR)))  # 저장소 루트
_DATA_DIR = os.path.join(_ROOT, "data")                                  # 모든 버전 공용 데이터


class NGramLM:
    # 버전마다 아래에서 바꾸는 설정 (기본값)
    ORDERS = [1]           # 만들 문맥 표의 길이 목록
    MAX_LENGTH = 20        # 문장을 최대 몇 토큰까지 만들지 (안전장치)

    def __init__(self):
        self.model = None

    # ---------- 토크나이저 (v0.0.1: 띄어쓰기) ----------
    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def tokenizer_name(self):
        return "whitespace"

    # ---------- 학습 ----------
    def read_sentences(self, path):
        sentences = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
        return sentences

    def prepare(self, tokens):
        """학습 전에 토큰을 손봅니다. (v0.0.1 은 그대로. v0.0.3 이 <END> 를 붙임)"""
        return tokens

    def count_into(self, table, tokens, n):
        """tokens 를 훑으며 '앞 n토큰 -> 다음 토큰' 개수를 table 에 더합니다."""
        for i in range(len(tokens) - n):
            context = " ".join(tokens[i:i + n])
            next_token = tokens[i + n]
            table.setdefault(context, {})
            table[context][next_token] = table[context].get(next_token, 0) + 1

    def train(self, sentences):
        """ORDERS 에 있는 길이마다 '문맥 -> 다음 토큰 개수' 표를 만듭니다."""
        tables = {str(n): {} for n in self.ORDERS}
        for sentence in sentences:
            tokens = self.prepare(self.tokenize(sentence))
            for n in self.ORDERS:
                self.count_into(tables[str(n)], tokens, n)
        return tables

    def to_dict(self, tables):
        return {
            "max_order": max(self.ORDERS),
            "tokenizer": self.tokenizer_name(),
            "tables": tables,
        }

    def save(self, tables, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(tables), f, ensure_ascii=False, indent=2)

    # ---------- 불러오기 ----------
    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.model = json.load(f)
        return self

    @classmethod
    def load_or_exit(cls, path):
        if not os.path.exists(path):
            raise SystemExit("모델 파일이 없어요! 먼저 train/train.py 를 실행해 주세요.")
        return cls().load(path)

    # ---------- 생성 ----------
    def _tables(self):
        return self.model["tables"]

    def _orders_desc(self):
        return sorted((int(k) for k in self._tables()), reverse=True)

    def choose(self, counts, temperature):
        """다음 토큰 하나를 고릅니다. (v0.0.1: 가장 많이 나온 것 = 그리디)"""
        candidates = list(counts.keys())
        nums = list(counts.values())
        return candidates[nums.index(max(nums))]

    def next_token(self, recent, temperature):
        """다음 토큰을 찾습니다. (v0.0.1: 가장 긴 문맥 표에서만 — 백오프 없음)"""
        n = self._orders_desc()[0]
        if len(recent) >= n:
            counts = self._tables()[str(n)].get(" ".join(recent[-n:]))
            if counts:
                return self.choose(counts, temperature)
        return None

    def can_continue(self, recent):
        n = self._orders_desc()[0]
        return len(recent) >= n and bool(self._tables()[str(n)].get(" ".join(recent[-n:])))

    def is_end(self, token):
        """문장을 끝낼 토큰인지. (v0.0.1: 문장끝 개념 없음. v0.0.3 이 <END> 를 추가)"""
        return False

    def generate(self, start_text, temperature=0.0):
        recent = self.tokenize(start_text)
        if not recent or not self.can_continue(recent):
            top_table = self._tables()[str(self._orders_desc()[0])]
            recent = random.choice(list(top_table.keys())).split()
        while len(recent) < self.MAX_LENGTH:
            nxt = self.next_token(recent, temperature)
            if nxt is None or self.is_end(nxt):
                break
            recent.append(nxt)
        return self.detokenize(recent)

    # ---------- 편의 실행기 ----------
    def run_train(self, data_path, model_path):
        print(f"데이터를 읽는 중... ({data_path})")
        sentences = self.read_sentences(data_path)
        print(f"문장 {len(sentences)}개 / 설정: 문맥{self.ORDERS}, 토크나이저={self.tokenizer_name()}")
        tables = self.train(sentences)
        for n in self.ORDERS:
            print(f"  표{n}: 서로 다른 문맥 {len(tables[str(n)])}개")
        self.save(tables, model_path)
        print(f"모델 저장 완료 -> {model_path}")


class Model(NGramLM):
    ORDERS = [1]              # 앞 1단어만 봄


DATA_PATH = os.path.join(_DATA_DIR, "pretrain", "train.txt")
MODEL_PATH = os.path.join(_HERE, "model.json")
