"""후보 재학습·비교·웹앱용 최선 GPT 내보내기."""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("v035_comparison", Path(__file__).resolve().parents[1] / "0.model/comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
if __name__ == "__main__":
    comparison.main()
