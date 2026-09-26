"""Frozen model identity, evaluation seals and inherited architecture contracts."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v046_tests',VERSION/'0.model/capstone.py')
c=importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
m=c.m


def model():
    lm=m.Model(); lm.DEVICE='cpu'; lm.EMBED=8; lm.HEADS=2; lm.LAYERS=2
    lm.FFN_HIDDEN=16; lm.BLOCK_SIZE=8; lm.VOCAB_SIZE=270; lm.VARIANT='rope'
    lm.DROPOUT=0; torch.manual_seed(17); lm.initialize(['abc 한글']); lm.net.eval()
    return lm


def identity(lm):
    return {'weights_sha256':c.weights_hash(lm.net),'tokenizer_semantic_sha256':c.tokenizer_hash(lm),
            'architecture':{k:getattr(lm,k) for k in c.ARCH_KEYS},
            'corpus_manifest_sha256':lm.corpus_manifest_sha256}


class CapstoneTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2)

    def test_semantic_fingerprint_reload_and_mutation(self):
        lm=model(); freeze=identity(lm)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'; lm.save(path)
            loaded=m.Model(); loaded.DEVICE='cpu'; loaded.load(path)
            c.validate_model(loaded,freeze)
            c.validate_sidecars(loaded,Path(d))
            self.assertEqual(lm.evaluate(['unseen']),loaded.evaluate(['unseen']))
            config_path=Path(d)/'config.json'
            import json
            config=json.loads(config_path.read_text()); config['block_size']+=1
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'exported'): c.validate_sidecars(loaded,Path(d))
            with torch.no_grad(): loaded.net.embedding.weight[7,0]+=1
            with self.assertRaisesRegex(ValueError,'mismatch'): c.validate_model(loaded,freeze)

    def test_architecture_and_tokenizer_mismatch(self):
        lm=model(); freeze=identity(lm)
        for key in ('tokenizer_semantic_sha256','corpus_manifest_sha256'):
            changed=copy.deepcopy(freeze); changed[key]='changed'
            with self.assertRaisesRegex(ValueError,'mismatch'): c.validate_model(lm,changed)
        changed=copy.deepcopy(freeze); changed['architecture']['BLOCK_SIZE']=9
        with self.assertRaisesRegex(ValueError,'mismatch'): c.validate_model(lm,changed)

    def test_independent_documents_reject_normalized_overlap_and_groups(self):
        old=[{'id':'old','text':'가  나','source':'s','topic':'old','template_id':'old'}]
        new={'id':'new','text':'unique','source':'final','topic':'new','template_id':'new'}
        c.check_independence([new],old)
        for field,value in [('text','가 나'),('topic','old'),('template_id','old')]:
            bad={**new,field:value}
            with self.assertRaisesRegex(ValueError,'overlap'): c.check_independence([bad],old)
        with self.assertRaises(ValueError): c.check_independence([new,new],old)
        with self.assertRaises(ValueError): c.check_independence([],old)

    def test_final_seal_rejects_modified_text_or_freeze(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); freeze=root/'freeze.json'; freeze.write_text('{}')
            records=[{'id':'unique','text':'sealed fixture text','source':'fixture',
                      'topic':'test-fixture-only','template_id':'test-fixture-only'}]
            source=root/'test.jsonl'
            # JSONL requires one JSON object per physical line.
            import json
            source.write_text(json.dumps(records[0])+'\n')
            manifest={'freeze_sha256':m.corpus.sha256(freeze.read_bytes()),
                      'file_sha256':m.corpus.sha256(source.read_bytes()),'documents':1}
            (root/'manifest.json').write_bytes(m.corpus.json_bytes(manifest))
            c.load_final_test(root,freeze)
            source.write_text(source.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'seal'): c.load_final_test(root,freeze)
            source.write_text(json.dumps(records[0])+'\n'); freeze.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'seal'): c.load_final_test(root,freeze)


if __name__=='__main__': unittest.main()
