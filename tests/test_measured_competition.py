"""Evidence gate: never promote CPU fixtures or missing GPU data to speedup."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

spec=importlib.util.spec_from_file_location("compare_measured_attn",ROOT/"scripts/compare_measured_attn.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MeasuredBoundaryEvidenceTests(unittest.TestCase):
    def test_refuse_no_gpu_and_no_complete_128_step_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"unavailable.json"
            p.write_text(json.dumps({
                "schema":module.GPU_SCHEMA,"status":"UNAVAILABLE_NO_CUDA_DEVICE",
                "hardware":{"cuda_available":False},
                "model_id":"meta-llama/Llama-3.1-8B","measurements":[]}))
            with self.assertRaisesRegex(ValueError,"no qualified measured"):
                module.validate_gpu(p,mode="graph")

            from cim_shared_prefix.profiles import PROFILES
            llama=PROFILES[0]
            p.write_text(json.dumps({
                "schema":module.GPU_SCHEMA,"status":"GPU_KERNEL_MEASUREMENTS_RECORDED",
                "hardware":{"cuda_available":True,"gpu_name":"unit-test-only","cuda_runtime":"test"},
                "model_id":llama.model_id,"model_revision":llama.revision,
                "precision":"bf16",
                "requested":{"batch":1,"prefix":2048,"suffix":128,
                             "output_tokens":128,"steps":[0]},
                "measurements":[{"token_step":0,"status":"MEASURED_ATTENTION_KERNEL_ONLY"}]}))
            with self.assertRaisesRegex(ValueError,"EVERY output step"):
                module.validate_gpu(p,mode="graph")

    def test_reject_wrong_model_and_precision(self):
        from cim_shared_prefix.profiles import PROFILES
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"unsupported.json"
            sample={"schema":module.GPU_SCHEMA,
                    "status":"GPU_KERNEL_MEASUREMENTS_RECORDED",
                    "hardware":{"cuda_available":True,"gpu_name":"fixture","cuda_runtime":"fixture"},
                    "model_id":PROFILES[0].model_id,
                    "model_revision":"wrong_revision","precision":"bf16",
                    "requested":{},
                    "measurements":[]}
            p.write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError,"revision"):
                module.validate_gpu(p,mode="graph")
            sample["model_revision"]=PROFILES[0].revision
            sample["precision"]="int4"
            p.write_text(json.dumps(sample))
            with self.assertRaises(ValueError):
                module.validate_gpu(p,mode="graph")


if __name__=="__main__":
    unittest.main()
