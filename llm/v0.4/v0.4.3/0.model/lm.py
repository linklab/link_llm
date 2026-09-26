"""Longer context, larger network and tied token input/output weights."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('v043_previous', HERE.parents[1]/'v0.4.2/0.model/lm.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
module, corpus, packing = previous.module, previous.corpus, previous.packing
ByteBPE, sequence_loss = previous.ByteBPE, previous.sequence_loss
Network = previous.previous.PackedNetwork


class NeuralLM(previous.NeuralLM):
    MODEL_VERSION = 'v0.4.3'
    BLOCK_SIZE, EMBED, LAYERS, FFN_HIDDEN, EPOCHS = 64, 96, 6, 384, 8
    TIE_WEIGHTS = True

    def build_net(self):
        net = super().build_net()
        if self.TIE_WEIGHTS:
            net.head.weight = net.embedding.weight
        return net

    def extra_metadata(self):
        return {**super().extra_metadata(), 'tie_weights': self.TIE_WEIGHTS}

    def restore_extra_metadata(self, meta):
        super().restore_extra_metadata(meta)
        if type(meta.get('tie_weights')) is not bool:
            raise ValueError('missing weight tying policy')
        self.TIE_WEIGHTS = meta['tie_weights']

    def training_config(self):
        return {**super().training_config(), 'TIE_WEIGHTS': self.TIE_WEIGHTS}


Model = NGramLM = NeuralLM
MODEL_PATH, VOCAB_PATH = str(HERE/'model.pt'), str(HERE/'vocab.json')
DATA_PATH, VALID_PATH = previous.DATA_PATH, previous.VALID_PATH
