"""Construct the SFT input-compatible network, without training or base loading."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def configuration(template):
    return {'format': 'sft-network-setup-v1', 'vocab_size': template.vocab_size,
            'template_sha256': template.fingerprint, 'block_size': 256,
            'embed': 64, 'layers': 4, 'heads': 4, 'ffn_hidden': 256,
            'dropout': 0.1, 'variant': 'rope', 'kv_heads': 2,
            'tie_weights': True, 'pad_id': 0, 'bos_id': 2, 'eos_id': 1,
            'eot_id': template.eot_id, 'stop_token_ids': list(template.stop_token_ids),
            'initialization': 'random-smoke-test-only',
            'base_checkpoint_loaded': False, 'trained': False}


def build_network(config, template):
    # Validate before importing torch; frozen settings are not a trainer config.
    if config != configuration(template):
        raise ValueError('model configuration does not match template/setup contract')
    spec = importlib.util.spec_from_file_location(
        'sft051_network', ROOT/'llm/v0.4/v0.4.5/0.model/lm.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.ModernNetwork(config['vocab_size'], config['embed'],
                               config['block_size'], config['heads'],
                               config['ffn_hidden'], config['dropout'],
                               config['layers'], config['variant'], config['kv_heads'])
    net.head.weight = net.embedding.weight
    return net
