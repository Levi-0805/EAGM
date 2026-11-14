"""Isolated postprocessing tests; excluded from frozen training source tree."""
import importlib.util,json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('ood',Path(__file__).with_name('evaluate_ood.py'));ood=importlib.util.module_from_spec(spec);spec.loader.exec_module(ood)
from tests.eagm2.test_extra_data import CountingEnv
from eagm2.models import ExecutionWorldModel

class OODTests(unittest.TestCase):
    def test_cap(self):
        env=CountingEnv();env.reset(1,0,.01);count=[0];wrapped=ood.CappedEnv(env,count,1)
        wrapped.step(np.zeros(14))
        with self.assertRaisesRegex(RuntimeError,'cap'):wrapped.step(np.zeros(14))
        self.assertEqual(env.calls,1)
    def test_checkpoint_load_and_complete_orchestration(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'reports/eagm2_dev') as tmp:
            root=Path(tmp);source=root/'development_v1';(source/'checkpoints').mkdir(parents=True)
            (source/'config.json').write_text(json.dumps(dict(task='tracking',seed=200001,smoke=False)))
            (source/'source_hashes.json').write_text('{}')
            stats=dict(xmean=np.zeros(28),xstd=np.ones(28),umean=np.zeros(14),ustd=np.ones(14),ymean=np.zeros(28),ystd=np.ones(28),lower=-np.ones(14),upper=np.ones(14),alpha=.5)
            for name in ood.METHODS:
                config=dict(obs_dim=28,width=160,memory='recurrent' if name=='combined_dir' else 'none',prior='original' if name=='ff_base' else 'delay')
                model=ExecutionWorldModel(stats,**config)
                torch.save(dict(stats=stats,model_config=config,state_dict=model.state_dict(),method=name,seed=200001),source/'checkpoints'/f'{name}.pt')
                torch.save(dict(theta=np.zeros(28)),source/'checkpoints'/f'{name}_policy.pt')
            self.assertEqual(tuple(ood.load_source(source)[2]),ood.METHODS)
            output=root/'output'
            with patch.object(sys,'argv',['evaluate_ood','--source',str(source),'--output',str(output)]),patch.object(ood,'make_env',side_effect=lambda task,cfg:CountingEnv()),patch.object(ood,'physical',return_value=-.5) as phys:
                ood.main()
                self.assertEqual(phys.call_count,48)
            result=json.loads((output/'results.json').read_text())
            self.assertEqual(len(result['rows']),36);self.assertTrue(result['source_unchanged'])
            budget=json.loads((output/'budget.json').read_text());self.assertEqual(budget['issued_control_steps'],4*3*96)
            for name in ood.SHIFTS:
                data=torch.load(output/f'{name}_observable_history_and_snapshots.pt',weights_only=False)
                self.assertEqual(len(data['starts']['validation']),3)
                self.assertFalse(data['starts']['train']);self.assertFalse(data['starts']['development_test'])
    def test_wrong_attempt_rejected(self):
        with self.assertRaises(ValueError):ood.load_source('/tmp/smoke_v1')
if __name__=='__main__':unittest.main(verbosity=2)
