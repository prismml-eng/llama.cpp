# Implementation Plan - Saved Before Context Compaction

## What I've Done
1. Read the entire Prism Launch Demo notebook
2. Cloned prism-llama-cpp fork (branch: prism)
3. Analyzed all CUDA kernels for Q1_0 and Q1_0_g128
4. Wrote optimization_report.md with detailed analysis
5. Wrote context_notes.md with all critical file locations and line numbers

## What I Still Need To Do

### Step 1: Modify vecdotq.cuh (MMVQ kernel - token generation)
File: prism-llama-cpp/ggml/src/ggml-cuda/vecdotq.cuh

**A) Add LUT at top of file (around line 108, before the VDR defines):**
```cuda
// Precomputed LUT: 4-bit nibble → packed int32 of four signed bytes {-1,+1}
// bit=0 → 0xFF (-1 as int8), bit=1 → 0x01 (+1 as int8)
__device__ static const int Q1_UNPACK_LUT[16] = {
    (int)0xFFFFFFFF, (int)0xFFFFFF01, (int)0xFFFF01FF, (int)0xFFFF0101,
    (int)0xFF01FFFF, (int)0xFF01FF01, (int)0xFF0101FF, (int)0xFF010101,
    (int)0x01FFFFFF, (int)0x01FFFF01, (int)0x01FF01FF, (int)0x01FF0101,
    (int)0x0101FFFF, (int)0x0101FF01, (int)0x010101FF, (int)0x01010101,
};
```

**B) Replace vec_dot_q1_0_q8_1_impl (lines 114-155):**
```cuda
template <int vdr> static __device__ __forceinline__ float vec_dot_q1_0_q8_1_impl(
    const int * v, const int * u, const float & d1, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = v[i];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int nibble = (vi >> (j * 4)) & 0x0F;
            sumi = ggml_cuda_dp4a(Q1_UNPACK_LUT[nibble], u[8*i + j], sumi);
        }
    }
    const float2 ds8f = __half22float2(ds8);
    return d1 * ds8f.x * sumi;
}
```

**C) Fix vec_dot_q1_0_q8_1 (lines 691-710) - use 32-bit load:**
Replace byte-by-byte load with memcpy or reinterpret_cast.

**D) Fix vec_dot_q1_0_g128_q8_1 (lines 712-750) - same LUT + 32-bit load:**
Replace the bit unpack loop with LUT lookup.

### Step 2: Modify mmq.cuh (MMQ kernel - prompt processing)
File: prism-llama-cpp/ggml/src/ggml-cuda/mmq.cuh

**A) In load_tiles_q1_0 (lines 302-374):**
Replace the bit unpack loop with LUT lookup (same pattern).

**B) In load_tiles_q1_0_g128 (lines 376-446):**
Same LUT optimization.

### Step 3: Modify dequantize.cuh
File: prism-llama-cpp/ggml/src/ggml-cuda/dequantize.cuh
Minor optimization: use single 32-bit load for qs bytes.

### Step 4: Create optimized notebook
Create a new notebook with:
- Profiling cells (nsys/ncu if available, or timing benchmarks)
- Optimized build instructions
- Performance comparison before/after
- Recommended runtime parameters for T4

## Compilation Command
```bash
cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cd llama.cpp && cmake --build build --config Release -j 32
```

## Key Constants
- Q1_UNPACK_LUT needs to be __device__ static const or in constant memory
- It's only 64 bytes (16 × 4) so fits in constant cache
- The LUT approach eliminates ALL branches and reduces register pressure
- dp4a(LUT[nibble], u, sumi) is the hot inner loop - should be ~3 instructions
