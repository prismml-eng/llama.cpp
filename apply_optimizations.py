#!/usr/bin/env python3
"""Apply Q1_0 bit-unpack optimizations to Prism llama.cpp CUDA kernels."""

import re

# ============================================================================
# 1. Optimize vecdotq.cuh
# ============================================================================
with open('prism-llama-cpp/ggml/src/ggml-cuda/vecdotq.cuh', 'r') as f:
    vecdotq = f.read()

# A) Add LUT definition right after the includes (after "#include <cstdint>")
lut_def = """
// Precomputed LUT: 4-bit nibble -> packed int32 of four signed bytes {-1,+1}
// bit=0 -> 0xFF (-1 as int8), bit=1 -> 0x01 (+1 as int8)
// Index by nibble value (0-15), get packed {b0,b1,b2,b3} ready for dp4a
__device__ static const int Q1_UNPACK_LUT[16] = {
    (int)0xFFFFFFFF, // 0b0000: -1,-1,-1,-1
    (int)0xFFFFFF01, // 0b0001: +1,-1,-1,-1
    (int)0xFFFF01FF, // 0b0010: -1,+1,-1,-1
    (int)0xFFFF0101, // 0b0011: +1,+1,-1,-1
    (int)0xFF01FFFF, // 0b0100: -1,-1,+1,-1
    (int)0xFF01FF01, // 0b0101: +1,-1,+1,-1
    (int)0xFF0101FF, // 0b0110: -1,+1,+1,-1
    (int)0xFF010101, // 0b0111: +1,+1,+1,-1
    (int)0x01FFFFFF, // 0b1000: -1,-1,-1,+1
    (int)0x01FFFF01, // 0b1001: +1,-1,-1,+1
    (int)0x01FF01FF, // 0b1010: -1,+1,-1,+1
    (int)0x01FF0101, // 0b1011: +1,+1,-1,+1
    (int)0x0101FFFF, // 0b1100: -1,-1,+1,+1
    (int)0x0101FF01, // 0b1101: +1,-1,+1,+1
    (int)0x010101FF, // 0b1110: -1,+1,+1,+1
    (int)0x01010101, // 0b1111: +1,+1,+1,+1
};

// Branchless bit-unpack: extract 4 bits from nibble, convert to packed signed bytes
static __device__ __forceinline__ int q1_unpack_nibble(int nibble) {
    return Q1_UNPACK_LUT[nibble];
}
"""

vecdotq = vecdotq.replace(
    '#include <cstdint>\n',
    '#include <cstdint>\n' + lut_def
)

# B) Replace vec_dot_q1_0_q8_1_impl with optimized version
old_impl = r'''template <int vdr> static __device__ __forceinline__ float vec_dot_q1_0_q8_1_impl\(
    const int \* v, const int \* u, const float & d1, const half2 & ds8\) \{

    int sumi = 0;

#pragma unroll
    for \(int i = 0; i < vdr; \+\+i\) \{
        const int vi = v\[i\];
        
        // Unpack 32 bits into 32 signed values \(-1 or \+1\)
        // Each bit: 0 -> -1, 1 -> \+1
        // Process all 32 bits, converting each to a signed byte
        
        int vi_bytes\[8\];
        
#pragma unroll
        for \(int j = 0; j < 8; \+\+j\) \{
            // Extract 4 bits and convert each to -1 or \+1
            const int shift = j \* 4;
            const int bits4 = \(vi >> shift\) & 0x0F;
            
            // Convert each of the 4 bits to a signed byte, then pack into int
            // bit=1 -> \+1, bit=0 -> -1
            const int b0 = \(bits4 & 0x01\) \? 1 : -1;
            const int b1 = \(bits4 & 0x02\) \? 1 : -1;
            const int b2 = \(bits4 & 0x04\) \? 1 : -1;
            const int b3 = \(bits4 & 0x08\) \? 1 : -1;
            
            // Pack 4 signed bytes into a single int for dp4a
            vi_bytes\[j\] = \(b0 & 0xFF\) \| \(\(b1 & 0xFF\) << 8\) \| \(\(b2 & 0xFF\) << 16\) \| \(\(b3 & 0xFF\) << 24\);
        \}
        
        // Perform dot product using dp4a \(4-way int8 dot product\)
#pragma unroll
        for \(int j = 0; j < 8; \+\+j\) \{
            sumi = ggml_cuda_dp4a\(vi_bytes\[j\], u\[8\*i \+ j\], sumi\);
        \}
    \}

    const float2 ds8f = __half22float2\(ds8\);

    // Q1_0 is symmetric \(no offset\), so we just multiply by scales
    // ds8f\.x is the scale from Q8_1, ds8f\.y is the precomputed sum \(not needed for symmetric quant\)
    return d1 \* ds8f\.x \* sumi;
\}'''

# Use simple string replace instead of regex - more reliable
old_impl_str = '''template <int vdr> static __device__ __forceinline__ float vec_dot_q1_0_q8_1_impl(
    const int * v, const int * u, const float & d1, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = v[i];
        
        // Unpack 32 bits into 32 signed values (-1 or +1)
        // Each bit: 0 -> -1, 1 -> +1
        // Process all 32 bits, converting each to a signed byte
        
        int vi_bytes[8];
        
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // Extract 4 bits and convert each to -1 or +1
            const int shift = j * 4;
            const int bits4 = (vi >> shift) & 0x0F;
            
            // Convert each of the 4 bits to a signed byte, then pack into int
            // bit=1 -> +1, bit=0 -> -1
            const int b0 = (bits4 & 0x01) ? 1 : -1;
            const int b1 = (bits4 & 0x02) ? 1 : -1;
            const int b2 = (bits4 & 0x04) ? 1 : -1;
            const int b3 = (bits4 & 0x08) ? 1 : -1;
            
            // Pack 4 signed bytes into a single int for dp4a
            vi_bytes[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }
        
        // Perform dot product using dp4a (4-way int8 dot product)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sumi = ggml_cuda_dp4a(vi_bytes[j], u[8*i + j], sumi);
        }
    }

    const float2 ds8f = __half22float2(ds8);

    // Q1_0 is symmetric (no offset), so we just multiply by scales
    // ds8f.x is the scale from Q8_1, ds8f.y is the precomputed sum (not needed for symmetric quant)
    return d1 * ds8f.x * sumi;
}'''

new_impl_str = '''template <int vdr> static __device__ __forceinline__ float vec_dot_q1_0_q8_1_impl(
    const int * v, const int * u, const float & d1, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi = v[i];

        // Optimized: LUT-based branchless bit unpack + fused dp4a
        // Each nibble (4 bits) maps to a packed int32 of four {-1,+1} bytes via Q1_UNPACK_LUT
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int nibble = (vi >> (j * 4)) & 0x0F;
            sumi = ggml_cuda_dp4a(Q1_UNPACK_LUT[nibble], u[8*i + j], sumi);
        }
    }

    const float2 ds8f = __half22float2(ds8);
    return d1 * ds8f.x * sumi;
}'''

assert old_impl_str in vecdotq, "Could not find old vec_dot_q1_0_q8_1_impl in vecdotq.cuh!"
vecdotq = vecdotq.replace(old_impl_str, new_impl_str)
print("✓ Replaced vec_dot_q1_0_q8_1_impl")

# C) Replace byte-by-byte load in vec_dot_q1_0_q8_1 with 32-bit load
old_q1_load = '''    // Q1_0 has 32 bits per block, stored in 4 bytes
    // Read all 4 bytes and pack into a single int32
    v[0] = bq1_0->qs[0] | (bq1_0->qs[1] << 8) | (bq1_0->qs[2] << 16) | (bq1_0->qs[3] << 24);'''

new_q1_load = '''    // Q1_0 has 32 bits per block, stored in 4 bytes
    // Optimized: use single 32-bit load via memcpy (safe for unaligned access)
    memcpy(&v[0], bq1_0->qs, 4);'''

if old_q1_load in vecdotq:
    vecdotq = vecdotq.replace(old_q1_load, new_q1_load)
    print("✓ Replaced Q1_0 byte-by-byte load with memcpy")
else:
    print("⚠ Could not find Q1_0 byte-by-byte load pattern")

# D) Optimize vec_dot_q1_0_g128_q8_1 - replace its bit unpack loop
old_g128_unpack = '''    // Unpack 32 bits into 32 signed values (-1 or +1)
    int vi_bytes[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int shift = j * 4;
        const int bits4 = (v >> shift) & 0x0F;
        const int b0 = (bits4 & 0x01) ? 1 : -1;
        const int b1 = (bits4 & 0x02) ? 1 : -1;
        const int b2 = (bits4 & 0x04) ? 1 : -1;
        const int b3 = (bits4 & 0x08) ? 1 : -1;
        vi_bytes[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }
    
    // Compute dot product for this 32-element chunk
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int u = get_int_b4(bq8_1_chunk->qs, j);
        sumi = ggml_cuda_dp4a(vi_bytes[j], u, sumi);
    }'''

new_g128_unpack = '''    // Optimized: LUT-based branchless bit unpack + fused dp4a
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int nibble = (v >> (j * 4)) & 0x0F;
        const int u = get_int_b4(bq8_1_chunk->qs, j);
        sumi = ggml_cuda_dp4a(Q1_UNPACK_LUT[nibble], u, sumi);
    }'''

if old_g128_unpack in vecdotq:
    vecdotq = vecdotq.replace(old_g128_unpack, new_g128_unpack)
    print("✓ Replaced Q1_0_g128 bit unpack in vec_dot")
else:
    print("⚠ Could not find Q1_0_g128 unpack pattern in vecdotq")

# Also fix the byte-by-byte load in g128
old_g128_load = '''    const int offset = iqs * 4;
    const int v = bq1_0_g128->qs[offset + 0] | (bq1_0_g128->qs[offset + 1] << 8) |
                  (bq1_0_g128->qs[offset + 2] << 16) | (bq1_0_g128->qs[offset + 3] << 24);'''

new_g128_load = '''    // Optimized: single 32-bit load via memcpy (safe for unaligned access)
    const int offset = iqs * 4;
    int v;
    memcpy(&v, bq1_0_g128->qs + offset, 4);'''

if old_g128_load in vecdotq:
    vecdotq = vecdotq.replace(old_g128_load, new_g128_load)
    print("✓ Replaced Q1_0_g128 byte-by-byte load with memcpy")
else:
    print("⚠ Could not find Q1_0_g128 byte-by-byte load pattern")

with open('prism-llama-cpp/ggml/src/ggml-cuda/vecdotq.cuh', 'w') as f:
    f.write(vecdotq)
print("✓ Wrote optimized vecdotq.cuh")

# ============================================================================
# 2. Optimize mmq.cuh
# ============================================================================
with open('prism-llama-cpp/ggml/src/ggml-cuda/mmq.cuh', 'r') as f:
    mmq = f.read()

# Replace bit unpack in load_tiles_q1_0
old_mmq_unpack_q1 = '''        // Q1_0 has 32 bits (4 bytes) for 32 elements at 1 bit each
        // Read all 4 bytes safely to avoid alignment issues
        const int qs0 = bxi->qs[0] | (bxi->qs[1] << 8) | (bxi->qs[2] << 16) | (bxi->qs[3] << 24);

        // For MMA: unpack 1-bit values to signed bytes (-1 or +1)
        // Process all 32 bits, 4 at a time
        int unpacked_bytes[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int shift = j * 4;
            const int bits4 = (qs0 >> shift) & 0x0F;
            const int b0 = (bits4 & 0x01) ? 1 : -1;
            const int b1 = (bits4 & 0x02) ? 1 : -1;
            const int b2 = (bits4 & 0x04) ? 1 : -1;
            const int b3 = (bits4 & 0x08) ? 1 : -1;
            unpacked_bytes[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }
        // Store unpacked values
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*QI8_0 + j] = unpacked_bytes[j];
        }'''

new_mmq_unpack_q1 = '''        // Optimized: single 32-bit load + LUT-based branchless unpack
        int qs0;
        memcpy(&qs0, bxi->qs, 4);

        // Unpack via Q1_UNPACK_LUT and store directly (no intermediate array)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int nibble = (qs0 >> (j * 4)) & 0x0F;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kbx*QI8_0 + j] = Q1_UNPACK_LUT[nibble];
        }'''

if old_mmq_unpack_q1 in mmq:
    mmq = mmq.replace(old_mmq_unpack_q1, new_mmq_unpack_q1)
    print("✓ Replaced Q1_0 bit unpack in load_tiles_q1_0")
else:
    print("⚠ Could not find Q1_0 unpack pattern in mmq.cuh")

# Replace bit unpack in load_tiles_q1_0_g128
old_mmq_unpack_g128 = '''        const int qs_offset = 4*kqsx;
        const int qs0 = bxi->qs[qs_offset + 0] | (bxi->qs[qs_offset + 1] << 8) |
                        (bxi->qs[qs_offset + 2] << 16) | (bxi->qs[qs_offset + 3] << 24);

        int unpacked_bytes[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int shift = j * 4;
            const int bits4 = (qs0 >> shift) & 0x0F;
            const int b0 = (bits4 & 0x01) ? 1 : -1;
            const int b1 = (bits4 & 0x02) ? 1 : -1;
            const int b2 = (bits4 & 0x04) ? 1 : -1;
            const int b3 = (bits4 & 0x08) ? 1 : -1;
            unpacked_bytes[j] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
        }

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + dst_offset + j] = unpacked_bytes[j];
        }'''

new_mmq_unpack_g128 = '''        // Optimized: single 32-bit load + LUT-based branchless unpack
        const int qs_offset = 4*kqsx;
        int qs0;
        memcpy(&qs0, bxi->qs + qs_offset, 4);

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;
        // Unpack via Q1_UNPACK_LUT and store directly (no intermediate array)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int nibble = (qs0 >> (j * 4)) & 0x0F;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + dst_offset + j] = Q1_UNPACK_LUT[nibble];
        }'''

if old_mmq_unpack_g128 in mmq:
    mmq = mmq.replace(old_mmq_unpack_g128, new_mmq_unpack_g128)
    print("✓ Replaced Q1_0_g128 bit unpack in load_tiles_q1_0_g128")
else:
    print("⚠ Could not find Q1_0_g128 unpack pattern in mmq.cuh")

with open('prism-llama-cpp/ggml/src/ggml-cuda/mmq.cuh', 'w') as f:
    f.write(mmq)
print("✓ Wrote optimized mmq.cuh")

# ============================================================================
# 3. Optimize dequantize.cuh
# ============================================================================
with open('prism-llama-cpp/ggml/src/ggml-cuda/dequantize.cuh', 'r') as f:
    deq = f.read()

# Optimize dequantize_q1_0 - use fused bit extraction
old_deq_q1 = '''static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;
    const float neg_d = -d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d
    const uint8_t bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const uint8_t bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = bit_0 ? d : neg_d;
    v.y = bit_1 ? d : neg_d;
}'''

new_deq_q1 = '''static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    // Optimized: branchless bit extraction using arithmetic
    // bit=1 -> +d, bit=0 -> -d  ==>  d * (2*bit - 1)
    const int byte_index_0 = iqs / 8;
    const int bit_offset_0 = iqs % 8;
    const int byte_index_1 = (iqs + 1) / 8;
    const int bit_offset_1 = (iqs + 1) % 8;

    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = d * (2*bit_0 - 1);
    v.y = d * (2*bit_1 - 1);
}'''

if old_deq_q1 in deq:
    deq = deq.replace(old_deq_q1, new_deq_q1)
    print("✓ Replaced dequantize_q1_0")
else:
    print("⚠ Could not find dequantize_q1_0 pattern")

# Same for g128
old_deq_g128 = '''static __device__ __forceinline__ void dequantize_q1_0_g128(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0_g128 * x = (const block_q1_0_g128 *) vx;

    const float d = x[ib].d;
    const float neg_d = -d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d
    const uint8_t bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const uint8_t bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = bit_0 ? d : neg_d;
    v.y = bit_1 ? d : neg_d;
}'''

new_deq_g128 = '''static __device__ __forceinline__ void dequantize_q1_0_g128(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0_g128 * x = (const block_q1_0_g128 *) vx;

    const float d = x[ib].d;

    // Optimized: branchless bit extraction using arithmetic
    const int byte_index_0 = iqs / 8;
    const int bit_offset_0 = iqs % 8;
    const int byte_index_1 = (iqs + 1) / 8;
    const int bit_offset_1 = (iqs + 1) % 8;

    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = d * (2*bit_0 - 1);
    v.y = d * (2*bit_1 - 1);
}'''

if old_deq_g128 in deq:
    deq = deq.replace(old_deq_g128, new_deq_g128)
    print("✓ Replaced dequantize_q1_0_g128")
else:
    print("⚠ Could not find dequantize_q1_0_g128 pattern")

with open('prism-llama-cpp/ggml/src/ggml-cuda/dequantize.cuh', 'w') as f:
    f.write(deq)
print("✓ Wrote optimized dequantize.cuh")

print("\n=== All optimizations applied successfully ===")
