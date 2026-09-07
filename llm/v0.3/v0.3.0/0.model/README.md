# 모델 코드 읽는 순서

1. `network.py`: Q/K/V와 인과 마스크, `(B,T,V)` 출력.
2. `sequences.py`: 한 칸 이동한 정답, overlap 손실 제외, 오른쪽 PAD.
3. `training.py`: 실제 토큰 수로 가중한 손실, Adam, 검증 최저 복원.
4. `lm.py`: 기존 토크나이저/웹앱 인터페이스와 새로운 시퀀스 엔진 연결.

설정·실행 명령·검증·한계는 [버전 README](../README.md)를 참고하세요.
