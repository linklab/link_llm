# -*- coding: utf-8 -*-
"""
base.py  (v0.0.1)  -  언어 모델의 '기반 클래스' NGramLM

여기에는 v0.0.1 이 처음으로 필요로 하는 함수들이 모두 들어 있어요.
(문장 읽기 / 토크나이저 / 학습 / 저장·불러오기 / 문장 만들기)

뒤 버전들(v0.0.2, v0.0.3 ...)은 이 클래스를 '상속' 해서,
자기 버전에서 새로 필요한 함수만 자기 폴더의 base.py 에 덧붙입니다.
그래서 "이 버전이 무엇을 더했나?" 를 그 버전의 base.py 만 보면 알 수 있어요.

[v0.0.1 의 성격]  앞 1단어 / 그리디(가장 많이 나온 단어) / 문장끝 없음 / 백오프 없음 / 띄어쓰기
"""

import json
import os
import random


class NGramLM:
    # 버전마다 model.py 에서 바꾸는 설정 (기본값)
    ORDERS = [1]           # 만들 문맥 표의 길이 목록
    MAX_LENGTH = 20        # 문장을 최대 몇 토큰까지 만들지 (안전장치)

    def __init__(self):
        self.model = None

    # ---------- 토크나이저 (v0.0.1: 띄어쓰기) ----------
    def tokenize(self, text):
        """문장을 토큰 리스트로 나눕니다."""
        return text.split()

    def detokenize(self, tokens):
        """토큰 리스트를 다시 문장으로 합칩니다."""
        return " ".join(tokens)

    def tokenizer_name(self):
        """이 모델이 쓰는 토크나이저 이름 (저장할 때 기록용)."""
        return "whitespace"

    # ---------- 학습 ----------
    def read_sentences(self, path):
        """파일에서 문장을 한 줄씩 읽어 리스트로 돌려줍니다."""
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
        """저장할 모델(딕셔너리) 모양. (모든 버전 공통 형식)"""
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
        """저장해둔 모델을 불러옵니다. (자기 자신을 돌려줌)"""
        with open(path, "r", encoding="utf-8") as f:
            self.model = json.load(f)
        return self

    @classmethod
    def load_or_exit(cls, path):
        """파일이 없으면 친절히 안내하고 멈춥니다. (test.py 에서 사용)"""
        if not os.path.exists(path):
            raise SystemExit("모델 파일이 없어요! 먼저 train/train.py 를 실행해 주세요.")
        return cls().load(path)

    # ---------- 생성 ----------
    def _tables(self):
        return self.model["tables"]

    def _orders_desc(self):
        """가진 표의 길이들을 큰 것부터 정렬. (예: [2, 1])"""
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
        """지금 토큰들로 이어갈 수 있는지. (v0.0.1: 가장 긴 문맥 표 기준)"""
        n = self._orders_desc()[0]
        return len(recent) >= n and bool(self._tables()[str(n)].get(" ".join(recent[-n:])))

    def is_end(self, token):
        """문장을 끝낼 토큰인지. (v0.0.1: 문장끝 개념 없음. v0.0.3 이 <END> 를 추가)"""
        return False

    def generate(self, start_text, temperature=0.0):
        """start_text 로 시작해 문장을 만듭니다."""
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
