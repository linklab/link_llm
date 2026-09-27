"""Generation must produce valid UTF-8 without deleting or replacing input text."""
import importlib.util
from pathlib import Path
import unittest
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('utf8_generation_tests',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ScriptedNetwork(torch.nn.Module):
    def __init__(self,vocab,choices):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(1))
        self.vocab,self.choices,self.step=vocab,choices,0

    def forward(self,tokens):
        logits=torch.full((1,tokens.shape[1],self.vocab),-1000.)
        for rank,token in enumerate(self.choices[min(self.step,len(self.choices)-1)]):
            logits[0,-1,token]=100.-rank*10
        self.step+=1
        return logits

    def cached(self,tokens,cache=None):
        return self(tokens)[:,-1],None


def scripted(byte_choices,limit):
    lm=m.Model(); lm.DEVICE='cpu'; lm.bpe=m.ByteBPE(); lm.MAX_LENGTH=limit
    # None means EOS; integers denote actual byte values, not vocabulary IDs.
    choices=[[1 if byte is None else byte+6 for byte in row] for row in byte_choices]
    lm.net=ScriptedNetwork(len(lm.bpe.tokens),choices)
    return lm


class UTF8Tests(unittest.TestCase):
    def test_all_single_bytes_at_character_boundary(self):
        grammar=m.UTF8Constraint(m.ByteBPE()); allowed=grammar.allowed(3)
        for byte in range(256):
            self.assertEqual(bool(allowed[byte+6]),byte<128 or 0xc2<=byte<=0xf4)
        self.assertTrue(allowed[1])
        self.assertFalse(allowed[[0,2,3,4,5]].any())

    def test_overlong_surrogate_and_out_of_range_rejected(self):
        for lead,low,high in ((0xe0,0xa0,0xbf),(0xed,0x80,0x9f),
                              (0xf0,0x90,0xbf),(0xf4,0x80,0x8f)):
            grammar=m.UTF8Constraint(m.ByteBPE()); grammar.accept(lead+6)
            allowed=grammar.allowed(3)
            self.assertFalse(allowed[1])
            for byte in range(256):
                self.assertEqual(bool(allowed[byte+6]),low<=byte<=high)

    def test_merged_pieces_can_cross_character_boundaries(self):
        bpe=m.ByteBPE()
        pieces=[b'\xea\xb0',b'\x80a\xf0',b'\x9f\x98\x80',b'\xe0\x80',b'a\xed\xa0\x80']
        bpe.pieces.extend(pieces); bpe._refresh()
        grammar=m.UTF8Constraint(bpe)
        ids=[262,263,264]
        for token in ids:
            self.assertTrue(grammar.allowed(3)[token]); grammar.accept(token)
        self.assertEqual(grammar.pending,b'')
        self.assertEqual(bpe.decode(ids),'가a😀')
        self.assertFalse(grammar.allowed(3)[265:].any())

    def test_invalid_logits_and_early_eos_are_masked_before_sampling(self):
        choices=[[0x80,0xea,ord('A')],[None,ord('A'),0xb0],[None,ord('A'),0x80],[None]]
        for cached in (False,True):
            for temperature in (0.,.8):
                lm=scripted(choices,4)
                self.assertEqual(lm.generate('아침 ',temperature,top_k=1,use_cache=cached,seed=9),'아침 가')
        lm=scripted(choices,4)
        self.assertIn('\ufffd',lm.generate('',valid_utf8=False))

    def test_limit_preserves_complete_korean_and_emoji(self):
        for text in ('가','😀'):
            raw=text.encode()
            choices=[[byte,ord('A')] for byte in raw]
            for limit in range(1,len(raw)+1):
                lm=scripted(choices,limit)
                result=lm.generate('')
                self.assertEqual(result,text if limit==len(raw) else 'A'*limit)
                result.encode().decode('utf-8',errors='strict')

    def test_prompt_is_preserved_including_literal_replacement_character(self):
        for prompt in ('',' ','한글 😀 가\n\ufffd'):
            self.assertEqual(scripted([[None]],4).generate(prompt),prompt)
            self.assertEqual(scripted([[None]],0).generate(prompt),prompt)


if __name__=='__main__': unittest.main()
