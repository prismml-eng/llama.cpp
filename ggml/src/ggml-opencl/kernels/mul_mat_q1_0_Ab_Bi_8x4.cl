// Q1_0 GEMM Kernel - Direct GGML layout (no transpose needed)
// Each work-item computes an 8x4 output tile
// gy indexes 8 output rows (N dimension - batch/sequence)
// gx indexes 4 output columns (M dimension - output features)
//
// Q1_0: 128 elements per block, 16 bytes (128 bits) + 1 half scale
// GGML stores B as N rows of K elements: B[n][k] at index n*K + k
// This kernel loads B values with strided access to avoid transpose

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifndef REQD_SUBGROUP_SIZE_128
#define REQD_SUBGROUP_SIZE_128
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_128
#endif

kernel void kernel_mul_mat_q1_0_Ab_Bi_8x4(
        global const uchar * src0_q,        // packed 1-bit weights (SOA: q buffer)
        global const half  * src0_d,        // scales (SOA: d buffer)
        global const uchar * src1_base,     // B activations base pointer
        ulong src1_offset,                  // offset into src1 buffer
        global uchar * dst_base,            // output base pointer
        ulong dst_offset,                   // offset into dst buffer
        int m,                              // M (output features / rows of A)
        int n,                              // N (batch size)
        int k,                              // K (input features / cols of A)
        int n_no_padding                    // N without padding (for bounds check)
) {
    // Apply offsets
    global const float * src1 = (global const float *)(src1_base + src1_offset);
    global float * dst = (global float *)(dst_base + dst_offset);

    int gy = get_global_id(0);  // output row tile (0 to N/8)
    int gx = get_global_id(1);  // output column tile (0 to M/4)
    int gx_4 = gx << 2;         // starting column (gx * 4)

    float8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;  // 8x4 output tile

    int num_blocks = k / 128;   // 128 elements per block for Q1_0
    int row_base = gy << 3;     // gy * 8 = starting output row

    // Pointers for 4 weight columns (SOA layout, row-major)
    // For Q1_0: each block is 16 bytes (128 bits)
    global const uchar* weight_base0 = src0_q + (gx_4 + 0) * num_blocks * 16;
    global const uchar* weight_base1 = src0_q + (gx_4 + 1) * num_blocks * 16;
    global const uchar* weight_base2 = src0_q + (gx_4 + 2) * num_blocks * 16;
    global const uchar* weight_base3 = src0_q + (gx_4 + 3) * num_blocks * 16;

    // Scale pointers for 4 columns
    global const half* scale_ptr0 = src0_d + (gx_4 + 0) * num_blocks;
    global const half* scale_ptr1 = src0_d + (gx_4 + 1) * num_blocks;
    global const half* scale_ptr2 = src0_d + (gx_4 + 2) * num_blocks;
    global const half* scale_ptr3 = src0_d + (gx_4 + 3) * num_blocks;

    for (int block = 0; block < num_blocks; block++) {
        // Load scales for 4 columns
        float s0 = (float)scale_ptr0[block];
        float s1 = (float)scale_ptr1[block];
        float s2 = (float)scale_ptr2[block];
        float s3 = (float)scale_ptr3[block];

        // Load 128 bits (4 uints) for each of 4 columns
        global const uint* bits_ptr0 = (global const uint*)(weight_base0 + block * 16);
        global const uint* bits_ptr1 = (global const uint*)(weight_base1 + block * 16);
        global const uint* bits_ptr2 = (global const uint*)(weight_base2 + block * 16);
        global const uint* bits_ptr3 = (global const uint*)(weight_base3 + block * 16);

        uint bits0_0 = bits_ptr0[0], bits0_1 = bits_ptr0[1], bits0_2 = bits_ptr0[2], bits0_3 = bits_ptr0[3];
        uint bits1_0 = bits_ptr1[0], bits1_1 = bits_ptr1[1], bits1_2 = bits_ptr1[2], bits1_3 = bits_ptr1[3];
        uint bits2_0 = bits_ptr2[0], bits2_1 = bits_ptr2[1], bits2_2 = bits_ptr2[2], bits2_3 = bits_ptr2[3];
        uint bits3_0 = bits_ptr3[0], bits3_1 = bits_ptr3[1], bits3_2 = bits_ptr3[2], bits3_3 = bits_ptr3[3];

        // Process 128 K elements in this block
        int k_base = block * 128;

        // Process first 32 bits (elements 0-31)
        #pragma unroll 4
        for (int i = 0; i < 32; i++) {
            int k_idx = k_base + i;

            // Load 8 B values for 8 output rows at K position k_idx
            float8 B;
            B.s0 = (row_base + 0 < n) ? src1[(row_base + 0) * k + k_idx] : 0.0f;
            B.s1 = (row_base + 1 < n) ? src1[(row_base + 1) * k + k_idx] : 0.0f;
            B.s2 = (row_base + 2 < n) ? src1[(row_base + 2) * k + k_idx] : 0.0f;
            B.s3 = (row_base + 3 < n) ? src1[(row_base + 3) * k + k_idx] : 0.0f;
            B.s4 = (row_base + 4 < n) ? src1[(row_base + 4) * k + k_idx] : 0.0f;
            B.s5 = (row_base + 5 < n) ? src1[(row_base + 5) * k + k_idx] : 0.0f;
            B.s6 = (row_base + 6 < n) ? src1[(row_base + 6) * k + k_idx] : 0.0f;
            B.s7 = (row_base + 7 < n) ? src1[(row_base + 7) * k + k_idx] : 0.0f;

            float w0 = ((bits0_0 >> i) & 1u) ? s0 : -s0;
            float w1 = ((bits1_0 >> i) & 1u) ? s1 : -s1;
            float w2 = ((bits2_0 >> i) & 1u) ? s2 : -s2;
            float w3 = ((bits3_0 >> i) & 1u) ? s3 : -s3;

            c0 += B * w0;
            c1 += B * w1;
            c2 += B * w2;
            c3 += B * w3;
        }

        // Process second 32 bits (elements 32-63)
        #pragma unroll 4
        for (int i = 0; i < 32; i++) {
            int k_idx = k_base + 32 + i;

            float8 B;
            B.s0 = (row_base + 0 < n) ? src1[(row_base + 0) * k + k_idx] : 0.0f;
            B.s1 = (row_base + 1 < n) ? src1[(row_base + 1) * k + k_idx] : 0.0f;
            B.s2 = (row_base + 2 < n) ? src1[(row_base + 2) * k + k_idx] : 0.0f;
            B.s3 = (row_base + 3 < n) ? src1[(row_base + 3) * k + k_idx] : 0.0f;
            B.s4 = (row_base + 4 < n) ? src1[(row_base + 4) * k + k_idx] : 0.0f;
            B.s5 = (row_base + 5 < n) ? src1[(row_base + 5) * k + k_idx] : 0.0f;
            B.s6 = (row_base + 6 < n) ? src1[(row_base + 6) * k + k_idx] : 0.0f;
            B.s7 = (row_base + 7 < n) ? src1[(row_base + 7) * k + k_idx] : 0.0f;

            float w0 = ((bits0_1 >> i) & 1u) ? s0 : -s0;
            float w1 = ((bits1_1 >> i) & 1u) ? s1 : -s1;
            float w2 = ((bits2_1 >> i) & 1u) ? s2 : -s2;
            float w3 = ((bits3_1 >> i) & 1u) ? s3 : -s3;

            c0 += B * w0;
            c1 += B * w1;
            c2 += B * w2;
            c3 += B * w3;
        }

        // Process third 32 bits (elements 64-95)
        #pragma unroll 4
        for (int i = 0; i < 32; i++) {
            int k_idx = k_base + 64 + i;

            float8 B;
            B.s0 = (row_base + 0 < n) ? src1[(row_base + 0) * k + k_idx] : 0.0f;
            B.s1 = (row_base + 1 < n) ? src1[(row_base + 1) * k + k_idx] : 0.0f;
            B.s2 = (row_base + 2 < n) ? src1[(row_base + 2) * k + k_idx] : 0.0f;
            B.s3 = (row_base + 3 < n) ? src1[(row_base + 3) * k + k_idx] : 0.0f;
            B.s4 = (row_base + 4 < n) ? src1[(row_base + 4) * k + k_idx] : 0.0f;
            B.s5 = (row_base + 5 < n) ? src1[(row_base + 5) * k + k_idx] : 0.0f;
            B.s6 = (row_base + 6 < n) ? src1[(row_base + 6) * k + k_idx] : 0.0f;
            B.s7 = (row_base + 7 < n) ? src1[(row_base + 7) * k + k_idx] : 0.0f;

            float w0 = ((bits0_2 >> i) & 1u) ? s0 : -s0;
            float w1 = ((bits1_2 >> i) & 1u) ? s1 : -s1;
            float w2 = ((bits2_2 >> i) & 1u) ? s2 : -s2;
            float w3 = ((bits3_2 >> i) & 1u) ? s3 : -s3;

            c0 += B * w0;
            c1 += B * w1;
            c2 += B * w2;
            c3 += B * w3;
        }

        // Process fourth 32 bits (elements 96-127)
        #pragma unroll 4
        for (int i = 0; i < 32; i++) {
            int k_idx = k_base + 96 + i;

            float8 B;
            B.s0 = (row_base + 0 < n) ? src1[(row_base + 0) * k + k_idx] : 0.0f;
            B.s1 = (row_base + 1 < n) ? src1[(row_base + 1) * k + k_idx] : 0.0f;
            B.s2 = (row_base + 2 < n) ? src1[(row_base + 2) * k + k_idx] : 0.0f;
            B.s3 = (row_base + 3 < n) ? src1[(row_base + 3) * k + k_idx] : 0.0f;
            B.s4 = (row_base + 4 < n) ? src1[(row_base + 4) * k + k_idx] : 0.0f;
            B.s5 = (row_base + 5 < n) ? src1[(row_base + 5) * k + k_idx] : 0.0f;
            B.s6 = (row_base + 6 < n) ? src1[(row_base + 6) * k + k_idx] : 0.0f;
            B.s7 = (row_base + 7 < n) ? src1[(row_base + 7) * k + k_idx] : 0.0f;

            float w0 = ((bits0_3 >> i) & 1u) ? s0 : -s0;
            float w1 = ((bits1_3 >> i) & 1u) ? s1 : -s1;
            float w2 = ((bits2_3 >> i) & 1u) ? s2 : -s2;
            float w3 = ((bits3_3 >> i) & 1u) ? s3 : -s3;

            c0 += B * w0;
            c1 += B * w1;
            c2 += B * w2;
            c3 += B * w3;
        }
    }

    // Write 8x4 tile to output
    if (row_base + 0 < n_no_padding) {
        vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + (row_base + 0) * m + (gx << 2));
    }
    if (row_base + 1 < n_no_padding) {
        vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + (row_base + 1) * m + (gx << 2));
    }
    if (row_base + 2 < n_no_padding) {
        vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + (row_base + 2) * m + (gx << 2));
    }
    if (row_base + 3 < n_no_padding) {
        vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + (row_base + 3) * m + (gx << 2));
    }
    if (row_base + 4 < n_no_padding) {
        vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + (row_base + 4) * m + (gx << 2));
    }
    if (row_base + 5 < n_no_padding) {
        vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + (row_base + 5) * m + (gx << 2));
    }
    if (row_base + 6 < n_no_padding) {
        vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + (row_base + 6) * m + (gx << 2));
    }
    if (row_base + 7 < n_no_padding) {
        vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + (row_base + 7) * m + (gx << 2));
    }
}
