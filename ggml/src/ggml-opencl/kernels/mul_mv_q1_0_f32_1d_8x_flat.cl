#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK1_0 128

typedef uchar uint8_t;
typedef ushort uint16_t;

// Based on working Q1_0 pattern - process 16 elements per thread
// Q1_0 has 128 elements per block, so we process 16 at a time
// il = 0,16,32,48,64,80,96,112 (8 different starting positions for 64 threads / 8 = 8 groups)
inline float block_q1_0_dot_y_flat(global uchar * x, global half * dh, float16 yl, int il) {
    float d = *dh;
    global ushort * qs = (global ushort *)x;
    
    // For 128-element block: 8 ushorts total (16 bytes)
    // Each thread processes 1 ushort (16 bits) based on il
    // il/16 gives us which ushort to read (0-7)
    uint bits = qs[il/16];
    
    int b0  = (bits >> 0) & 1u;
    int b1  = (bits >> 1) & 1u;
    int b2  = (bits >> 2) & 1u;
    int b3  = (bits >> 3) & 1u;
    int b4  = (bits >> 4) & 1u;
    int b5  = (bits >> 5) & 1u;
    int b6  = (bits >> 6) & 1u;
    int b7  = (bits >> 7) & 1u;
    int b8  = (bits >> 8) & 1u;
    int b9  = (bits >> 9) & 1u;
    int ba = (bits >> 10) & 1u;
    int bb = (bits >> 11) & 1u;
    int bc = (bits >> 12) & 1u;
    int bd = (bits >> 13) & 1u;
    int be = (bits >> 14) & 1u;
    int bf = (bits >> 15) & 1u;
    
    float s0  = (float)(b0  * 2 - 1);
    float s1  = (float)(b1  * 2 - 1);
    float s2  = (float)(b2  * 2 - 1);
    float s3  = (float)(b3  * 2 - 1);
    float s4  = (float)(b4  * 2 - 1);
    float s5  = (float)(b5  * 2 - 1);
    float s6  = (float)(b6  * 2 - 1);
    float s7  = (float)(b7  * 2 - 1);
    float s8  = (float)(b8  * 2 - 1);
    float s9  = (float)(b9  * 2 - 1);
    float sa = (float)(ba * 2 - 1);
    float sb = (float)(bb * 2 - 1);
    float sc = (float)(bc * 2 - 1);
    float sd = (float)(bd * 2 - 1);
    float se = (float)(be * 2 - 1);
    float sf = (float)(bf * 2 - 1);
    
    float acc = 0.f;
    acc += yl.s0 * s0;
    acc += yl.s1 * s1;
    acc += yl.s2 * s2;
    acc += yl.s3 * s3;
    acc += yl.s4 * s4;
    acc += yl.s5 * s5;
    acc += yl.s6 * s6;
    acc += yl.s7 * s7;
    acc += yl.s8 * s8;
    acc += yl.s9 * s9;
    acc += yl.sa * sa;
    acc += yl.sb * sb;
    acc += yl.sc * sc;
    acc += yl.sd * sd;
    acc += yl.se * se;
    acc += yl.sf * sf;
    
    return d * acc;
}

#define N_DST 8
#define N_SIMDGROUP 1
#ifdef cl_intel_required_subgroup_size
#define N_SIMDWIDTH 16
#else
#define N_SIMDWIDTH 64
#endif

inline void mul_vec_q1_0_f32_8x_flat(global uchar * src0_q, global half * src0_d, global float * src1, global float * dst, int ne00, int ne01, int ne02, int ne10, int ne12, int ne0, int ne1, int r2, int r3) {
    const ulong nb = ne00 / QK1_0;
    int r0 = get_group_id(0), r1 = get_group_id(1), im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;
    int i12 = im % ne12, i13 = im / ne12;
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * (QK1_0/8);
    global uchar * x = src0_q + offset0_q;
    global half * d = src0_d + offset0_d;
    global float * y = src1 + r1*ne10 + im*ne00*ne1;
    float16 yl; float8 sumf = 0.f;
    
    // For 128-element blocks: 64 threads / 8 groups = 8 threads per group
    // Each thread processes 16 elements (128/8 = 16)
    int ix = get_sub_group_local_id() / 8, il = 16 * (get_sub_group_local_id() % 8);
    global float * yb = y + ix * QK1_0 + il;
    
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        yl.s0=yb[0]; yl.s1=yb[1]; yl.s2=yb[2]; yl.s3=yb[3]; yl.s4=yb[4]; yl.s5=yb[5]; yl.s6=yb[6]; yl.s7=yb[7];
        yl.s8=yb[8]; yl.s9=yb[9]; yl.sa=yb[10]; yl.sb=yb[11]; yl.sc=yb[12]; yl.sd=yb[13]; yl.se=yb[14]; yl.sf=yb[15];
        sumf.s0 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 0*nb*(QK1_0/8), d + ib + 0*nb, yl, il);
        sumf.s1 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 1*nb*(QK1_0/8), d + ib + 1*nb, yl, il);
        sumf.s2 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 2*nb*(QK1_0/8), d + ib + 2*nb, yl, il);
        sumf.s3 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 3*nb*(QK1_0/8), d + ib + 3*nb, yl, il);
        sumf.s4 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 4*nb*(QK1_0/8), d + ib + 4*nb, yl, il);
        sumf.s5 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 5*nb*(QK1_0/8), d + ib + 5*nb, yl, il);
        sumf.s6 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 6*nb*(QK1_0/8), d + ib + 6*nb, yl, il);
        sumf.s7 += block_q1_0_dot_y_flat(x + ib*(QK1_0/8) + 7*nb*(QK1_0/8), d + ib + 7*nb, yl, il);
        yb += QK1_0 * (N_SIMDWIDTH/8);
    }
    float8 tot = (float8)(sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1), sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3), sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5), sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7));
    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        if (first_row + 1 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        if (first_row + 2 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        if (first_row + 3 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        if (first_row + 4 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        if (first_row + 5 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        if (first_row + 6 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        if (first_row + 7 < ne01) dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
    }
}

#ifdef cl_intel_required_subgroup_size
REQD_SUBGROUP_SIZE_16
#elif defined(cl_qcom_reqd_sub_group_size)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q1_0_f32_1d_8x_flat(global uchar * src0_q, global half * src0_d, global float * src1, ulong offset1, global float * dst, ulong offsetd, int ne00, int ne01, int ne02, int ne10, int ne12, int ne0, int ne1, int r2, int r3) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);
    mul_vec_q1_0_f32_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}
