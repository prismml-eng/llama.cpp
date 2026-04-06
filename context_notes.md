# Critical Context Notes

## Repo Structure
- Prism llama.cpp fork cloned to `prism-llama-cpp/`
- Branch: `prism`
- Original notebook: `Prism_Launch_Demo_(Colab).ipynb`

## Key Files to Modify
1. `prism-llama-cpp/ggml/src/ggml-cuda/vecdotq.cuh`
   - Lines 114-155: `vec_dot_q1_0_q8_1_impl` - MAIN TARGET for MMVQ optimization
   - Lines 691-710: `vec_dot_q1_0_q8_1` - loads bytes individually (4 byte loads → 1 int load)
   - Lines 712-750: `vec_dot_q1_0_g128_q8_1` - same issue
   - Lines 108-112: VDR defines (VDR_Q1_0_Q8_1_MMVQ=1, VDR_Q1_0_g128_Q8_1_MMVQ=1)

2. `prism-llama-cpp/ggml/src/ggml-cuda/mmq.cuh`
   - Lines 302-374: `load_tiles_q1_0` - same bit-unpack issue
   - Lines 376-446: `load_tiles_q1_0_g128` - same issue
   - Line 14: MMQ_ITER_K_Q1_0=128 (unused currently, kernel uses MMQ_ITER_K=256)

3. `prism-llama-cpp/ggml/src/ggml-cuda/dequantize.cuh`
   - Lines 1-50: dequantize_q1_0 and dequantize_q1_0_g128

## Block Structures (ggml/src/ggml-common.h)
- block_q1_0: { ggml_half d; uint8_t qs[4]; } = 6 bytes for 32 elements
- block_q1_0_g128: { ggml_half d; uint8_t qs[16]; } = 18 bytes for 128 elements
- QK1_0=32, QK1_0_g128=128
- QI1_0=1 (QK1_0/32), QI1_0_g128=4 (QK1_0_g128/32)

## T4 GPU Info
- sm_75 Turing, TURING_MMA_AVAILABLE defined
- For MMVQ: uses MMVQ_PARAMETERS_GENERIC table, nwarps=4 for ncols_dst=1
- For MMQ: uses Turing MMA path (tensor cores)
- dp4a available, no __vsub4 intrinsic on CUDA (need alternative)

## Key Optimization: Bit Unpack
Current: 32 branches per 32-bit word, 8 iterations with per-bit ternary
Optimized: Use multiply-scatter trick: (nibble * 0x08040201) & 0x01010101 to spread bits
Then convert {0,1} → {0xFF,0x01} using arithmetic (need byte-level sub)

IMPORTANT: __vsub4 is NOT a standard CUDA intrinsic. Use __vsubss4 or manual approach:
- Method: spread = (nibble * 0x08040201) & 0x01010101; packed = spread | (spread - 0x01010101)
- Wait, that borrows across bytes. Better: result = spread * 0x02 - 0x01010101 with byte saturation
- Actually simplest correct approach: use LUT or use the fact that:
  0→0xFF, 1→0x01 can be computed as: (spread * 0xFE) + 0x01010101 ... no
  Actually: byte = (2*bit - 1) as signed = bit ? 1 : -1
  In unsigned bytes: 0→0xFF, 1→0x01
  So: result = spread ^ 0x00000000 when bit=1, result = 0xFFFFFFFF when bit=0
  = NOT(NOT(spread) & mask_for_zeros) ... complicated
  
BEST APPROACH: Just use the 16-entry LUT in constant memory. 16 entries × 4 bytes = 64 bytes, fits in const cache.
Or even simpler: __byte_perm can reorganize bytes on Turing.

ACTUALLY SIMPLEST: The current code's ternary WILL compile to predicated moves (no real branches) on CUDA.
The REAL win is:
1. Eliminating the intermediate vi_bytes array (fuse unpack+dp4a)
2. Using the multiply-scatter trick instead of shift+mask+test for each bit
3. Fixing the byte-level memory loads

For byte subtraction across byte lanes, the correct CUDA intrinsic is:
- __vsubss4(a, b) for saturating signed sub
- Or just do: result = (spread << 1) | ~(spread | (spread << 1)) ... hmm
- Cleanest: result = __byte_perm(spread, 0x01FFFFFF, 0x7610) ... complex

Let me use the LUT approach - it's clearest and fastest (16 entries in constant cache):

```cuda
__device__ static const int Q1_LUT[16] = {
    (int)0xFFFFFFFF, // 0000: -1,-1,-1,-1
    (int)0xFFFFFF01, // 0001: +1,-1,-1,-1  
    (int)0xFFFF01FF, // 0010: -1,+1,-1,-1
    (int)0xFFFF0101, // 0011: +1,+1,-1,-1
    (int)0xFF01FFFF, // 0100: -1,-1,+1,-1
    (int)0xFF01FF01, // 0101: +1,-1,+1,-1
    (int)0xFF0101FF, // 0110: -1,+1,+1,-1
    (int)0xFF010101, // 0111: +1,+1,+1,-1
    (int)0x01FFFFFF, // 1000: -1,-1,-1,+1
    (int)0x01FFFF01, // 1001: +1,-1,-1,+1
    (int)0x01FF01FF, // 1010: -1,+1,-1,+1
    (int)0x01FF0101, // 1011: +1,+1,-1,+1
    (int)0x0101FFFF, // 1100: -1,-1,+1,+1
    (int)0x0101FF01, // 1101: +1,-1,+1,+1
    (int)0x010101FF, // 1110: -1,+1,+1,+1
    (int)0x01010101, // 1111: +1,+1,+1,+1
};
```

## Benchmark Baseline
- 8B Q1_0_g128 on T4: pp512=1302 t/s, tg128=59.4 t/s
- Model size: 1.07 GiB for 8.19B params
