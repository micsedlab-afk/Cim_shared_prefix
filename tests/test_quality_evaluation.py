import copy
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cim_shared_prefix.quality_evaluation import (
    fixed_windows,aggregate_units,validate_units,write_aggregate,token_digest,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        # Complete paired 8-sample evaluated units (explicit TEST FIXTURE ONLY).
        for lp in [256,512]:
            for j in range(8):
                for k,arm in enumerate(['R0','R1','R2','R3']):
                    # R3 may improve on R2, and this must remain admissible.
                    offsets=[0.,.02,.03,-.01]
                    mean=1.2+j*.1+(.2 if lp==256 else 0)+offsets[k]
                    self.rows.append(dict(model_id='test-only-model',arm=arm,batch=1,
                                          prefix_tokens=lp,suffix_tokens=4,
                                          token_count=4,nll_sum=4*mean,
                                          sample_id=f'eval_{j:03d}',targets_sha256=f'{j:064x}'))

    def test_windows_disjoint_and_token_hashed(self):
        stream=list(range(5000))
        a=fixed_windows(stream,model_id='toy',prefixes=(100,200),suffix=4)
        b=fixed_windows(stream,model_id='toy',prefixes=(100,200),suffix=4)
        self.assertEqual(a,b)
        self.assertEqual(len(a['development']),4)
        self.assertEqual(len(a['evaluation']),16)
        self.assertFalse(set(a['development_blocks']) & set(a['evaluation_blocks']))
        anchors={}
        for r in a['evaluation']:
            anchors.setdefault(r['sample_id'],set()).add(r['end'])
        self.assertTrue(all(len(x)==1 for x in anchors.values()))
        self.assertEqual(len(token_digest(stream)),64)

    def test_pausing_or_missing_window_fails_closed(self):
        for rows in [self.rows[:-1]]:
            with self.assertRaises(ValueError):
                validate_units(rows,model_id='test-only-model',prefixes=(256,512),suffix=4)
        rows=copy.deepcopy(self.rows)
        rows[0]['targets_sha256']='f'*64
        with self.assertRaises(ValueError):
            validate_units(rows,model_id='test-only-model',prefixes=(256,512),suffix=4)
        rows=copy.deepcopy(self.rows)
        rows[0]['model_id']='different-model'
        with self.assertRaises(ValueError):
            validate_units(rows,model_id='test-only-model',prefixes=(256,512),suffix=4)

    def test_deterministic_pair_bootstrap_and_negative_r3_effect(self):
        a=aggregate_units(self.rows,model_id='test-only-model',prefixes=(256,512),
                          suffix=4,resamples=1000)
        b=aggregate_units(list(reversed(self.rows)),model_id='test-only-model',
                          prefixes=(256,512),suffix=4,resamples=1000)
        self.assertEqual(a,b)
        es=[x for x in a['effects'] if x['effect']=='characterized_response']
        self.assertEqual(len(es),2)
        self.assertTrue(all(x['delta_nll']<0 for x in es))
        self.assertTrue(all(x['delta_nll_ci95_high']<0 for x in es))
        for row in a['absolute']:
            self.assertAlmostEqual(row['ppl'],math.exp(row['mean_nll']))
        with TemporaryDirectory() as root:
            write_aggregate(a,root,{'source':'fixture'})
            for f in ['quality.json','absolute_quality.csv','paired_effects.csv','fig_quality_effects.svg']:
                self.assertTrue((Path(root)/f).exists())

if __name__=='__main__':unittest.main()
