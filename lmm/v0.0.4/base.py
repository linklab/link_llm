# -*- coding: utf-8 -*-
"""
base.py  (v0.0.4)  -  백오프(backoff): 긴 문맥이 없으면 짧은 문맥으로 줄여 재시도

[이 버전이 새로 더한 것]
  - next_token(): 다음 토큰을 찾을 때, 긴 문맥(2단어)부터 시도하고
                  없으면 짧은 문맥(1단어)으로 줄여서 다시 찾음  (부모 것을 덮어씀)
  - can_continue(): 위와 같은 방식(백오프)으로 '이어갈 수 있는지' 판단  (부모 것을 덮어씀)

이게 되려면 표1·표2 가 둘 다 있어야 하므로, model.py 에서 ORDERS = [1, 2] 로 둡니다.
덕분에 단어를 1개만 입력해도 그 단어로 시작하는 문장을 만들 수 있어요.
"""

import os
import importlib.util


def _inherit(prev_version):
    lmm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(lmm_dir, prev_version, "base.py")
    spec = importlib.util.spec_from_file_location("lmmbase_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_inherit("v0.0.3")):
    def next_token(self, recent, temperature):
        """백오프: 긴 문맥부터 찾고, 없으면 짧은 문맥으로 줄여서 다시 찾습니다."""
        for n in self._orders_desc():          # 예) 2 -> 1
            if len(recent) >= n:
                counts = self._tables()[str(n)].get(" ".join(recent[-n:]))
                if counts:
                    return self.choose(counts, temperature)
        return None

    def can_continue(self, recent):
        """백오프까지 고려해, 이어갈 다음 토큰이 하나라도 있는지 확인합니다."""
        for n in self._orders_desc():
            if len(recent) >= n and self._tables()[str(n)].get(" ".join(recent[-n:])):
                return True
        return False
