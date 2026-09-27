"""Local training is portable; strict reproduction still checks the reference runtime."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v046_training_modes',VERSION/'1.train/train.py')
t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


class TrainingModesTests(unittest.TestCase):
    def test_local_accepts_source_changes_but_strict_rejects(self):
        frozen=json.loads((VERSION/'0.model/freeze.json').read_text())['source_hashes']
        actual=dict(frozen); actual[next(iter(actual))]='changed-by-local-experiment'
        with patch.object(t.c,'source_hashes',return_value=actual):
            self.assertEqual(t.check_sources(frozen,False),actual)
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                t.check_sources(frozen,True)

    def test_local_settings_are_applied_and_strict_recipe_is_preserved(self):
        frozen={'training_config':{'EPOCHS':4,'PATIENCE':0,'LR':.003}}
        with patch.object(t,'PATIENCE',3),patch.object(t,'EPOCHS',12):
            self.assertEqual(t.training_config(frozen,False),{'EPOCHS':12,'PATIENCE':3,'LR':.003})
            self.assertEqual(t.training_config(frozen,True),frozen['training_config'])
        self.assertEqual(t.training_config(frozen,False,16,2)['PATIENCE'],2)
        for epochs,patience in [(16,None),(None,3)]:
            with self.assertRaises(ValueError): t.training_config(frozen,True,epochs,patience)
        with self.assertRaises(ValueError): t.training_config(frozen,False,4,-1)
        self.assertEqual(frozen['training_config']['PATIENCE'],0)

    def test_local_mode_accepts_different_runtime_and_reports_it(self):
        recorded={'python':'3.12.14','torch':'2.13.0'}
        actual={'python':'3.12.10','torch':'2.10.0'}
        differences=t.check_runtime(recorded,actual,False)
        self.assertEqual(len(differences),2)
        self.assertIn('current=2.10.0',differences[1])

    def test_strict_mode_rejects_mismatch_with_actionable_versions(self):
        recorded={'python':'3.12.14','torch':'2.13.0'}
        with self.assertRaisesRegex(ValueError,'recorded=2.13.0, current=2.10.0'):
            t.check_runtime(recorded,{'python':'3.12.14','torch':'2.10.0'},True)
        self.assertEqual(t.check_runtime(recorded,recorded,True),[])

    def test_default_local_output_is_separate_and_strict_is_temporary(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(t.output_directory(None,False,d),t.DEFAULT_LOCAL_OUTPUT.resolve())
            self.assertEqual(t.output_directory(None,True,d),Path(d).resolve())
            self.assertEqual(t.output_directory(Path(d),False,'unused'),Path(d).resolve())

    def test_local_training_cannot_overwrite_a_frozen_model(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): t.output_directory(VERSION/'0.model',False,d)
            (Path(d)/'freeze.json').write_text('{}')
            with self.assertRaises(ValueError): t.output_directory(Path(d),False,'unused')
            self.assertEqual(t.output_directory(VERSION/'0.model',True,d),(VERSION/'0.model').resolve())


if __name__=='__main__': unittest.main()
