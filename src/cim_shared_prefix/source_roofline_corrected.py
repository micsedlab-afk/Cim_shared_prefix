"""Byte-correct LLMCompass A100 FP16 grouped GPU source roofline.

Matmul.__call__ stores io_count as *elements*, while the upstream
Matmul.roofline_model divides that element count by bandwidth in bytes/s.
This adapter preserves its historical source result and computes separately
the correct FP16 byte-based HBM and L2 source models.

Constants copied directly from vendored LLMCompass compute_module.py,
io_module.py and softmax.py in micsedlab-afk/MAGMA-gpt.
These remain source-model estimates, not measured CUDA kernels.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from math import isclose

A100_HBM_BPS=2039e9
A100_L2_BPS=5120*1.41e9
A100_MATMUL_FLOPS_PER_S=108*4*16*16*2*1.41e9
A100_VECTOR_FLOPS_PER_S=108*512*1.41e9
FP16_WORD_BYTES=2
SOFTMAX_MODEL_FLOPS_PER_ELEMENT=3*35+7


@dataclass(frozen=True)
class GPUContextSources:
    batch: int
    native_legacy_matmul_qk_s: float
    corrected_hbm_qk_s: float
    corrected_l2_qk_s: float
    corrected_hbm_pv_s: float
    corrected_l2_pv_s: float
    hbm_softmax_s: float
    l2_softmax_s: float
    hbm_unfused_full_prefix_s: float
    l2_unfused_full_prefix_s: float
    logical_qk_io_elements: int
    correct_fp16_qk_io_bytes: int
    status: str = "native_source_geometry_byte_corrected_HBM_vs_L2; not CUDA measurement"

    def json_record(self):
        return asdict(self)


def from_native_grouped_row(row: dict) -> GPUContextSources:
    """Exactly reconstruct grouped source operators at P2048 Q32 KV8 D128."""
    B=row.get("batch")
    if type(B) is not int or B<1:
        raise ValueError("invalid native GPU source batch")
    p=2048
    d=128
    g=4
    hk=8
    m=B*g
    expected_elements=hk*(m*d+d*p+m*p)
    expected_flops=hk*(2*m*d*p)
    if (row.get("KV_heads")!=hk or row.get("queries_per_KV_head")!=m
        or row.get("head_dim")!=d or row.get("prefix")!=p
        or row.get("native_operand_and_score_bytes")!=expected_elements
        or row.get("native_compute_flops")!=expected_flops):
        raise ValueError("pinned GQA source geometry or Matmul IO evidence mismatch")
    source=float(row["total_source_grouped_QK_s"])
    legacy=max(expected_flops/A100_MATMUL_FLOPS_PER_S,
               expected_elements/min(A100_HBM_BPS,A100_L2_BPS))
    if not isclose(source,legacy,rel_tol=1e-12,abs_tol=1e-16):
        raise ValueError("legacy source roofline unexpectedly differs")
    correct_bytes=expected_elements*FP16_WORD_BYTES
    def mm(bw):
        return max(expected_flops/A100_MATMUL_FLOPS_PER_S,correct_bytes/bw)
    rows=B*hk*g
    softmax_elements=rows*p
    softmax_io_bytes=softmax_elements*FP16_WORD_BYTES*3
    softmax_flops=softmax_elements*SOFTMAX_MODEL_FLOPS_PER_ELEMENT
    def softmax(bw):
        return max(softmax_flops/A100_VECTOR_FLOPS_PER_S,softmax_io_bytes/bw)
    # Source Matmul PV has M=B*G, K=P, N=D. Since QK exchanges
    # K and N, it has exactly the same FLOPs and IO element counts.
    hqk=mm(min(A100_HBM_BPS,A100_L2_BPS))
    lqk=mm(A100_L2_BPS)
    hs=softmax(min(A100_HBM_BPS,A100_L2_BPS))
    ls=softmax(A100_L2_BPS)
    return GPUContextSources(
        batch=B,native_legacy_matmul_qk_s=source,
        corrected_hbm_qk_s=hqk,corrected_l2_qk_s=lqk,
        corrected_hbm_pv_s=hqk,corrected_l2_pv_s=lqk,
        hbm_softmax_s=hs,l2_softmax_s=ls,
        hbm_unfused_full_prefix_s=2*hqk+hs,
        l2_unfused_full_prefix_s=2*lqk+ls,
        logical_qk_io_elements=expected_elements,
        correct_fp16_qk_io_bytes=correct_bytes,
    )
