# -*- coding: utf-8 -*-
"""
server.py  -  나만의 작은 언어 모델(LMM)을 브라우저에서 써보는 웹 서버

[무엇을 하나요?]
- 웹 브라우저로 http://127.0.0.1:8000 에 접속하면 ChatGPT 처럼 생긴 화면이 나와요.
- 화면 위쪽에서 '버전'(v0.0.1 등)을 고를 수 있어요.
- 메시지를 보내면, 그 버전이 학습해둔 모델(model.json)을 이용해 대답(문장)을 만들어 줍니다.

[어려운 설치가 필요 없어요]
- 파이썬에 기본으로 들어있는 http.server 만 사용해요. (pip install 필요 없음!)

[전체 그림]
브라우저 ──(메시지 + 버전)──▶ 이 서버 ──▶ 해당 버전의 model.json 을 읽어 문장 생성 ──▶ 브라우저에 대답 표시
"""

import json
import os
import inspect
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ----------------------------------------------------------------------
# 경로 설정
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))     # web_service 폴더
PROJECT = os.path.dirname(HERE)                        # link_lmm 폴더
LMM_DIR = os.path.join(PROJECT, "lmm")                 # 버전 폴더들이 모여있는 곳
INDEX_PATH = os.path.join(HERE, "index.html")         # 화면(HTML) 파일


# ----------------------------------------------------------------------
# 1) 버전 찾기: lmm 폴더 안에서 "v로 시작하고 2.models/model.json 이 있는" 폴더를 모읍니다.
# ----------------------------------------------------------------------
def find_versions():
    versions = []
    if not os.path.isdir(LMM_DIR):
        return versions
    for name in os.listdir(LMM_DIR):
        folder = os.path.join(LMM_DIR, name)
        model_file = os.path.join(folder, "2.models", "model.json")
        if name.startswith("v") and os.path.isdir(folder) and os.path.exists(model_file):
            versions.append(name)
    versions.sort()               # v0.0.1, v0.0.2 ... 순서대로 정렬
    return versions


def load_version_class(version):
    """
    선택한 버전 폴더의 2.models/lm.py 를 불러와, 그 버전의 NGramLM 클래스를 돌려줍니다.
    (폴더 이름에 '.' 이 있어 보통의 import 가 안 되므로 파일 경로로 불러와요.
     각 버전의 lm.py 는 이전 버전 lm.py 를 물려받는 사슬 구조예요.)
    """
    model_py = os.path.join(LMM_DIR, version, "2.models", "lm.py")
    spec = importlib.util.spec_from_file_location("lmmlm_" + version.replace(".", "_"), model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NGramLM


def load_lm(version):
    """선택한 버전의 클래스로 모델(model.json)을 불러옵니다. (각 버전 test.py 와 똑같은 클래스)"""
    cls = load_version_class(version)
    model_file = os.path.join(LMM_DIR, version, "2.models", "model.json")
    return cls().load(model_file)


# ----------------------------------------------------------------------
# 2) 문장 생성
# ----------------------------------------------------------------------
# 문장 생성 코드는 각 버전 폴더의 2.models/lm.py 에 있는 NGramLM 에 있어요.
# 웹서버는 선택된 버전의 클래스를 불러와서(load) generate 를 부를 뿐입니다.
def generate_reply(lm, message, temperature, top_k=0, top_p=1.0):
    """
    NGramLM 으로 대답 문장을 만듭니다.
    top_k / top_p 는 v0.0.7 부터 지원해요. 그 인자를 받는 버전에만 안전하게 넘깁니다.
    (구버전 generate 는 temperature 만 받으므로, 지원 여부를 보고 전달)
    """
    params = inspect.signature(lm.generate).parameters
    extra = {}
    if "top_k" in params:
        extra["top_k"] = top_k
    if "top_p" in params:
        extra["top_p"] = top_p
    return lm.generate(message, temperature, **extra)


# ----------------------------------------------------------------------
# 3) 웹 서버: 브라우저의 요청을 처리합니다.
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # GET 요청 처리 (화면 보여주기, 버전 목록 주기)
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            with open(INDEX_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/versions":
            # 브라우저가 "버전 목록 알려줘" 하고 물어보면 대답해줍니다.
            self._send_json({"versions": find_versions()})

        else:
            self._send_json({"error": "not found"}, status=404)

    # POST 요청 처리 (채팅 메시지 받아서 대답 만들기)
    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json({"error": "not found"}, status=404)
            return

        # 브라우저가 보낸 내용(JSON)을 읽어요.
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "잘못된 요청이에요."}, status=400)
            return

        version = data.get("version", "")
        message = data.get("message", "").strip()

        # 온도(temperature): 없으면 1.0, 이상한 값이 오면 안전한 범위(0~2)로 맞춤
        try:
            temperature = float(data.get("temperature", 1.0))
        except (TypeError, ValueError):
            temperature = 1.0
        temperature = max(0.0, min(2.0, temperature))

        # top-k (0 = 끄기), top-p (1.0 = 끄기). v0.0.7 부터 효과 있음
        try:
            top_k = int(data.get("top_k", 0))
        except (TypeError, ValueError):
            top_k = 0
        top_k = max(0, min(100, top_k))
        try:
            top_p = float(data.get("top_p", 1.0))
        except (TypeError, ValueError):
            top_p = 1.0
        top_p = max(0.0, min(1.0, top_p))

        if version not in find_versions():
            self._send_json({"error": "그런 버전은 없어요."}, status=400)
            return
        if not message:
            self._send_json({"error": "메시지를 입력해 주세요."}, status=400)
            return

        # 대화 기록: [[사용자말, 봇답], ...] (문자열 짝만 안전하게 걸러냄)
        history = []
        for pair in data.get("history", []):
            if isinstance(pair, list) and len(pair) >= 2 \
                    and isinstance(pair[0], str) and isinstance(pair[1], str):
                history.append((pair[0], pair[1]))

        lm = load_lm(version)
        if hasattr(lm, "chat"):
            # v0.0.8~ : 대화 형식 + 멀티턴 (기록을 문맥으로 삼아 봇의 답만 생성)
            reply = lm.chat(message, history, temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            # v0.0.7 이하 : 예전처럼 한 문장씩 이어 붙임
            reply = generate_reply(lm, message, temperature, top_k, top_p)
        self._send_json({"reply": reply, "version": version})

    # 서버 로그를 조용하게 (터미널을 깔끔하게 유지)
    def log_message(self, fmt, *args):
        pass


def main():
    host, port = "127.0.0.1", 8000
    server = ThreadingHTTPServer((host, port), Handler)
    print("=" * 50)
    print("  나만의 작은 언어 모델 웹앱이 켜졌어요!")
    print(f"  브라우저에서 http://{host}:{port} 로 접속하세요.")
    print("  끄려면 이 터미널에서 Ctrl + C 를 누르세요.")
    print("=" * 50)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료했어요. 안녕히 가세요!")
        server.server_close()


if __name__ == "__main__":
    main()
