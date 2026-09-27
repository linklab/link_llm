"""Scale-up experiment on the v0.4.6 architecture: bigger network, longer training budget.

Not a frozen capstone like v0.4.6 — defaults here are just a larger starting point;
1.train/train.py exposes the same CLI knobs as v0.4.5 so size/epochs stay adjustable.
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('v047_previous', HERE.parents[1]/'v0.4.6/0.model/lm.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
module, corpus, packing = previous.module, previous.corpus, previous.packing
ByteBPE, sequence_loss = previous.ByteBPE, previous.sequence_loss


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION = 'v0.4.7'
    VARIANT = 'rope'
    EMBED, LAYERS, HEADS, FFN_HIDDEN = 128, 6, 8, 512
    EPOCHS, PATIENCE = 60, 10


Model = NGramLM = NeuralLM
MODEL_PATH, VOCAB_PATH = str(HERE/'model.pt'), str(HERE/'vocab.json')
DATA_PATH, VALID_PATH = previous.DATA_PATH, previous.VALID_PATH
