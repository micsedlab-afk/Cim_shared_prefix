#!/usr/bin/env python3
"""Native source LLMCompass operator run with exact Llama-3.1-8B GQA dimensions."""
import argparse,json,sys
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cim_shared_prefix.contracts import Workload
from cim_shared_prefix.gqa_native import run_native_fp16,gqa_shapes

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--magma-root',type=Path,required=True);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--prefix',type=int,default=1024);p.add_argument('--suffix',type=int,default=128)
    p.add_argument('--output-tokens',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();w=Workload(a.batch,a.prefix,a.suffix,32,8,128,32,a.output_tokens)
    data={'schema':'cim-shared-prefix-gqa-native-operator-series/v1','workload':asdict(w),
          'points':[run_native_fp16(w,i,a.magma_root) for i in range(w.output_tokens)]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'output':str(a.output),'profiles':len(data['points']),
                      'status':'PARTIAL_NATIVE_FP16_GQA; COMPLETE_TPOT_UNSUPPORTED'}))
