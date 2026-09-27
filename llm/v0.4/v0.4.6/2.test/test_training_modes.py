"""Local training is portable; strict reproduction still checks the reference runtime."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v046_training_modes',VERSION/'1.train/train.py')
t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)


class TrainingModesTests(unittest.TestCase):
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
