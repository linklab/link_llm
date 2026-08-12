# -*- coding: utf-8 -*-
"""
base.py  (v0.0.2)  -  앞 '단어 2개' 로 예측

[이 버전이 새로 더한 함수]  없어요! 🎉

v0.0.1 의 코드가 이미 'ORDERS 에 적힌 길이'로 문맥을 세도록 만들어져 있어서,
앞 2단어를 보려면 model.py 에서 ORDERS = [2] 로 바꾸기만 하면 됩니다.
(어떤 업그레이드는 새 코드 없이 '설정' 만으로 되기도 해요.)

그래서 이 파일은 v0.0.1 의 NGramLM 을 그대로 물려받기만 합니다.
"""

import os
import importlib.util


def _inherit(prev_version):
    """이전 버전 폴더의 base.py 를 불러와 그 NGramLM 을 돌려줍니다.
    (폴더 이름에 '.' 이 있어 보통의 import 가 안 되므로 파일 경로로 불러와요.)"""
    lmm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(lmm_dir, prev_version, "base.py")
    spec = importlib.util.spec_from_file_location("lmmbase_" + prev_version.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


class NGramLM(_inherit("v0.0.1")):
    pass   # v0.0.1 과 똑같은 코드. 달라지는 건 model.py 의 ORDERS = [2] 뿐!
