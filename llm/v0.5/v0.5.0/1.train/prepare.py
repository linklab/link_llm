"""v0.5.0 prepares SFT data only; model training belongs to v0.5.3."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('sft050', ROOT/'data/sft/v0.5.0/dataset.py')
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)

if __name__ == '__main__':
    dataset.main()
