# -*- coding: utf-8 -*-
"""
dataloader.py  (v0.1.1)  -  PyTorch Dataset 정의 + DataLoader 생성

강의 자료(03.real_world_data_to_tensors)의 관례대로:
  - `Dataset` 을 상속해 __init__ / __len__ / __getitem__ 세 가지 정의,
    __getitem__ 은 한 샘플을 딕셔너리 {'input': ..., 'target': ...} 로 반환.
  - `DataLoader(dataset=..., batch_size=..., shuffle=...)` 로 감싸 배치·셔플을 맡김.

(어휘 만들기·짝 만들기는 0.model/lm.py 의 build_vocab / make_pairs 가 담당하고,
 여기서는 이미 텐서로 만든 (앞 토큰, 다음 토큰) 짝을 받아 감싸기만 해요.)
"""

try:
    from torch.utils.data import Dataset, DataLoader
except ImportError:          # torch 없이도 이 파일이 로딩되게
    Dataset = object
    DataLoader = None


class BigramDataset(Dataset):
    """
    (앞 토큰, 다음 토큰) 짝을 담는 데이터셋.
    __getitem__ 은 강의 자료 관례대로 딕셔너리 {'input': 앞 토큰, 'target': 다음 토큰} 을 돌려줘요.
    """
    def __init__(self, xs, ys):
        self.data = xs        # 앞 토큰 인덱스 (입력)
        self.target = ys      # 다음 토큰 인덱스 (정답)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input": self.data[idx], "target": self.target[idx]}


def build_dataloader(xs, ys, batch_size, shuffle=True):
    """
    (앞, 다음) 텐서를 BigramDataset 으로 감싸고 DataLoader 로 만들어 돌려줘요.
    학습 쪽(lm.py)에서는 이 한 줄만 부르면 됩니다:
        loader = build_dataloader(xs, ys, batch_size, shuffle=True)
    """
    dataset = BigramDataset(xs, ys)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
