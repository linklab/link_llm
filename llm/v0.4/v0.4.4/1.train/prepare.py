"""Re-export the v0.4.3 weights unchanged: caching is an inference-only change."""
import importlib.util
import json
from pathlib import Path
import torch

VERSION=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v044_prepare',VERSION/'0.model/lm.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    torch.set_num_threads(4)
    old=m.previous.Model(); old.DEVICE='cpu'; old.load(m.previous.MODEL_PATH)
    lm=m.Model(); lm.DEVICE='cpu'; lm.BLOCK_SIZE=old.BLOCK_SIZE; lm.EMBED=old.EMBED
    lm.itos=old.itos; lm.stoi=old.stoi; lm.best_epoch=old.best_epoch
    lm.restore_extra_metadata(old.extra_metadata())
    lm.net=lm.build_net(); lm.net.load_state_dict(old.net.state_dict()); lm.net.eval()
    for k,v in lm.net.state_dict().items():
        torch.testing.assert_close(v,old.net.state_dict()[k],rtol=0,atol=0)
    output=VERSION/'0.model'; lm.save(output/'model.pt')
    report=json.loads(Path(m.previous.MODEL_PATH).with_name('training_report.json').read_text())
    report.update(version=lm.MODEL_VERSION,source_checkpoint_sha256=report['checkpoint_sha256'],
                  checkpoint_sha256=m.corpus.sha256((output/'model.pt').read_bytes()),
                  preparation='Exact weight copy from v0.4.3; no new training. Training config/timing inherited.')
    (output/'training_report.json').write_bytes(m.corpus.json_bytes(report))
    print('Exact v0.4.3 weight copy saved as v0.4.4')


if __name__=='__main__': main()
