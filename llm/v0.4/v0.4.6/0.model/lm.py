"""Frozen GPT base capstone; architecture selected exclusively by v0.4.5 validation."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('v046_previous',HERE.parents[1]/'v0.4.5/0.model/lm.py')
previous=importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
module,corpus,packing=previous.module,previous.corpus,previous.packing
ByteBPE,sequence_loss=previous.ByteBPE,previous.sequence_loss


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION='v0.4.6'


Model=NGramLM=NeuralLM
MODEL_PATH,VOCAB_PATH=str(HERE/'model.pt'),str(HERE/'vocab.json')
DATA_PATH,VALID_PATH=previous.DATA_PATH,previous.VALID_PATH
