// Hand-rolled PTX twin of the CuTe Hopper wgmma path in mmq-hopper-q1.cu — same algorithm
// (dequant-in-SMEM + int8 wgmma, per-128-K activation scales, fixed-grid + stream-K), but the
// MMA layer is inline wgmma PTX + hand-built SMEM descriptors instead of CUTLASS/CuTe, so it
// builds with no external dependency. Selected at runtime with GGML_HOPPER_Q1_PTX=1.
//
// DRAFT: not yet run on sm_90a hardware. Acceptance gate before replacing the CuTe path:
//   1. bit-exact C vs the CuTe build (same instruction + accumulation order => identical s32)
//   2. test-backend-ops MUL_MAT vs CPU reference
// The quant/repack/cache helpers are duplicated from mmq-hopper-q1.cu on purpose (separate
// namespace, no shared header) — dedupe only after this path wins the A/B.
#include "common.cuh"

#include <unordered_map>

#if defined(GGML_USE_HOPPER_Q1_PTX)
#    include <cuda_pipeline.h>

namespace hopper_q1_ptx {

static constexpr int bM = 128, bN = 128, bK = 128;

// ---- SMEM layout: the same K-major 128-byte-swizzle GMMA layout CuTe's
// Layout_K_SW128_Atom<int8_t> produces, written out by hand. The atom is an 8-row x 128-byte
// core matrix; a 128-row operand tile is 16 core matrices stacked contiguously (1024 B apart).
// Within a core matrix, byte (r, c) lives at offset r*128 + (c ^ ((r%8) << 4)) — the swizzle
// XORs the row index into bits [6:4] of the column so the 16-byte chunks of a column land in
// distinct SMEM banks.
__device__ __forceinline__ int swz128_offset(int r, int c) {
    return (r >> 3) * 1024 + (r & 7) * 128 + (c ^ ((r & 7) << 4));
}

// ---- GMMA shared-memory matrix descriptor (PTX ISA: "asynchronous warpgroup matrix
// descriptor"). Field layout matches CUTLASS's GmmaDescriptor:
//   [13:0]  start address  (SMEM byte address >> 4)
//   [29:16] leading byte offset >> 4 (ignored for swizzled K-major layouts)
//   [45:32] stride byte offset  >> 4 (core-matrix stride: 8 rows * 128 B = 1024 B)
//   [51:49] base offset ((addr % 1024) >> 7, covers sub-1024-B base misalignment)
//   [63:62] swizzle mode (1 = 128-byte)
__device__ __forceinline__ uint64_t make_smem_desc_sw128(const int8_t * smem_ptr) {
    const uint32_t addr = (uint32_t) __cvta_generic_to_shared(smem_ptr);
    uint64_t       desc = 0;
    desc |= (uint64_t) ((addr >> 4) & 0x3FFF);
    desc |= (uint64_t) (1024 >> 4) << 32;
    desc |= (uint64_t) ((addr >> 7) & 0x7) << 49;
    desc |= (uint64_t) 1 << 62;
    return desc;
}

// advancing K by 32 bytes within a swizzled tile = +32 B on the (row-0) start address
__device__ __forceinline__ uint64_t desc_k_step(uint64_t desc, int kk) {
    return desc + (uint64_t) ((kk * 32) >> 4);
}

// ---- wgmma.mma_async m64n64k32 s32 += s8 * s8^T, both operands K-major in SMEM.
// scale_d = 0 zero-initializes the accumulator (saves the explicit clear on the first k-chunk).
template <int SCALE_D>
__device__ __forceinline__ void wgmma_m64n64k32_s8(int32_t (&d)[32], uint64_t desc_a, uint64_t desc_b) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, %34;\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]),
          "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),
          "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), "+r"(d[24]),
          "+r"(d[25]), "+r"(d[26]), "+r"(d[27]), "+r"(d[28]), "+r"(d[29]), "+r"(d[30]), "+r"(d[31])
        : "l"(desc_a), "l"(desc_b), "n"(SCALE_D));
}

__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;");
}

__device__ __forceinline__ void wgmma_wait0() {
    asm volatile("wgmma.wait_group.sync.aligned 0;");
}

// generic-proxy SMEM stores (our unpack) must be made visible to the async proxy (wgmma reads)
__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async.shared::cta;");
}

// ---- accumulator fragment coordinates (PTX ISA wgmma .m64nN .s32 layout). Each warpgroup
// thread holds 32 s32 values; warp w owns rows [16w, 16w+16). Same tiling as mma m16n8k32:
// regs come in groups of 4 per 8-column block — {row, row+8} x {col, col+1}.
__device__ __forceinline__ void acc_coord(int e, int warp, int lane, int & row, int & col) {
    row = warp * 16 + (lane >> 2) + 8 * ((e & 3) >> 1);
    col = (e >> 2) * 8 + (lane & 3) * 2 + (e & 1);
}

// fp32 -> int8 with per-128 absmax scale (same as the CuTe path; see mmq-hopper-q1.cu)
__global__ void quant_act_per128(const float * __restrict__ x,
                                 int8_t * __restrict__ q,
                                 float * __restrict__ d,
                                 int M,
                                 int K) {
    const int ngroups = M * (K / 128);
    const int g       = blockIdx.x * 8 + threadIdx.x / 32;
    if (g >= ngroups) {
        return;
    }
    const int      lane = threadIdx.x % 32;
    const int      m = g / (K / 128), kc = g % (K / 128);
    const float4 * xs   = reinterpret_cast<const float4 *>(x + (size_t) m * K + kc * 128) + lane;
    float4         v    = *xs;
    float          amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#    pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    const float scale = amax / 127.0f;
    const float inv   = scale > 0.f ? 1.0f / scale : 0.f;
    char4       out   = make_char4((char) lrintf(v.x * inv), (char) lrintf(v.y * inv), (char) lrintf(v.z * inv),
                                   (char) lrintf(v.w * inv));
    *(reinterpret_cast<char4 *>(q + (size_t) m * K + kc * 128) + lane) = out;
    if (lane == 0) {
        d[(size_t) m * (K / 128) + kc] = scale;
    }
}

__global__ void repack_q1_dense(const block_q1_0 * __restrict__ W,
                                unsigned * __restrict__ bits,
                                float * __restrict__ dw,
                                long nblocks_total) {
    long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks_total) {
        return;
    }
    const uint16_t * u16 = reinterpret_cast<const uint16_t *>(W + b);
    dw[b]                = __half2float(*reinterpret_cast<const __half *>(W + b));
#    pragma unroll
    for (int w = 0; w < 4; ++w) {
        bits[b * 4 + w] = (unsigned) u16[1 + 2 * w] | ((unsigned) u16[2 + 2 * w] << 16);
    }
}

__global__ void repack_q2_dense(const block_q2_0 * __restrict__ W,
                                unsigned * __restrict__ bits,
                                float * __restrict__ dw,
                                long nblocks_total) {
    long b = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks_total) {
        return;
    }
    const uint16_t * u16 = reinterpret_cast<const uint16_t *>(W + b);
    dw[b]                = __half2float(*reinterpret_cast<const __half *>(W + b));
#    pragma unroll
    for (int w = 0; w < 8; ++w) {
        bits[b * 8 + w] = (unsigned) u16[1 + 2 * w] | ((unsigned) u16[2 + 2 * w] << 16);
    }
}

struct DenseW {
    unsigned * bits;
    float *    dw;
};

static DenseW get_dense_w(const void * wdata, long N, long K, int wbits, cudaStream_t stream) {
    static std::unordered_map<const void *, DenseW> cache;
    auto                                            it = cache.find(wdata);
    if (it != cache.end()) {
        return it->second;
    }
    DenseW     d{};
    const long nb = N * (K / 128);
    cudaMalloc(&d.bits, nb * 16 * wbits);
    cudaMalloc(&d.dw, nb * sizeof(float));
    if (wbits == 1) {
        repack_q1_dense<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>((const block_q1_0 *) wdata, d.bits, d.dw,
                                                                           nb);
    } else {
        repack_q2_dense<<<(unsigned) ((nb + 255) / 256), 256, 0, stream>>>((const block_q2_0 *) wdata, d.bits, d.dw,
                                                                           nb);
    }
    cache.emplace(wdata, d);
    return d;
}

// SMEM plan (identical sizes to the CuTe path): double-buffered A (128x128 s8) + B (128x128 s8)
// + per-row/col scales. Operand tiles are written through swz128_offset so the descriptors match.
static constexpr int SMEM_TILE  = bM * bK;  // == bN * bK
static constexpr int SMEM_BYTES = 4 * SMEM_TILE + 2 * (bM + bN) * (int) sizeof(float);

// loads one k-chunk into SMEM buffer `buf`: A via cp.async, B unpacked from dense bit words.
// Mirrors the CuTe path's load_stage; see mmq-hopper-q1.cu for the LUT/entropy and cp.async notes.
template <int WBITS>
__device__ __forceinline__ void load_stage_ptx(int8_t * sA,
                                               int8_t * sB,
                                               float *  sDa,
                                               float *  sDw,
                                               const int8_t * __restrict__ Aq,
                                               const float * __restrict__ dA,
                                               const unsigned * __restrict__ Wbits,
                                               const float * __restrict__ Wd,
                                               int mblk,
                                               int nblk,
                                               int kc,
                                               int K) {
    const int     nblocks_row = K / 128;
    // A: 16-byte cp.async chunks; destination address is the swizzled offset of (r, c16)
    constexpr int A_CHUNKS    = bM * (bK / 16);
    for (int i = threadIdx.x; i < A_CHUNKS; i += 512) {
        int r = i / (bK / 16), c16 = (i % (bK / 16)) * 16;
        __pipeline_memcpy_async(sA + swz128_offset(r, c16), Aq + (size_t) (mblk * bM + r) * K + kc * bK + c16, 16);
    }
    __pipeline_commit();
    if constexpr (WBITS == 1) {
        constexpr int B_WORDS = bN * (bK / 32);
        for (int i = threadIdx.x; i < B_WORDS; i += 512) {
            int      r = i / (bK / 32), w = i % (bK / 32);
            unsigned bits = Wbits[((size_t) (nblk * bN + r) * nblocks_row + kc) * 4 + w];
            unsigned out[8];
#    pragma unroll
            for (int nib = 0; nib < 8; ++nib) {
                unsigned nb     = (bits >> (nib * 4)) & 0xF;
                unsigned spread = (nb & 1u) | ((nb & 2u) << 7) | ((nb & 4u) << 14) | ((nb & 8u) << 21);
                out[nib]        = __vadd4(0xFFFFFFFFu, spread << 1);
            }
            *reinterpret_cast<int4 *>(sB + swz128_offset(r, w * 32))      = make_int4(out[0], out[1], out[2], out[3]);
            *reinterpret_cast<int4 *>(sB + swz128_offset(r, w * 32 + 16)) = make_int4(out[4], out[5], out[6], out[7]);
        }
    } else {
        constexpr int B_WORDS = bN * (bK / 16);
        for (int i = threadIdx.x; i < B_WORDS; i += 512) {
            int      r = i / (bK / 16), w = i % (bK / 16);
            unsigned bits = Wbits[((size_t) (nblk * bN + r) * nblocks_row + kc) * 8 + w];
            unsigned out[4];
#    pragma unroll
            for (int b8 = 0; b8 < 4; ++b8) {
                unsigned f      = (bits >> (b8 * 8)) & 0xFFu;
                unsigned spread = (f & 0x03u) | ((f & 0x0Cu) << 6) | ((f & 0x30u) << 12) | ((f & 0xC0u) << 18);
                out[b8]         = __vsub4(spread, 0x01010101u);
            }
            *reinterpret_cast<int4 *>(sB + swz128_offset(r, w * 16)) = make_int4(out[0], out[1], out[2], out[3]);
        }
    }
    for (int i = threadIdx.x; i < bM; i += 512) {
        sDa[i] = dA[(size_t) (mblk * bM + i) * nblocks_row + kc];
    }
    for (int i = threadIdx.x; i < bN; i += 512) {
        sDw[i] = Wd[(size_t) (nblk * bN + i) * nblocks_row + kc];
    }
}

// runs the 4 k=32 wgmma sub-chunks of one 128-deep stage into acc (zero-initialized via scale_d)
__device__ __forceinline__ void mma_stage(int32_t (&acc)[32], const int8_t * sAh, const int8_t * sBh) {
    const uint64_t descA = make_smem_desc_sw128(sAh);
    const uint64_t descB = make_smem_desc_sw128(sBh);
    wgmma_fence();
    wgmma_m64n64k32_s8<0>(acc, descA, descB);
#    pragma unroll
    for (int kk = 1; kk < bK / 32; ++kk) {
        wgmma_m64n64k32_s8<1>(acc, desc_k_step(descA, kk), desc_k_step(descB, kk));
    }
    wgmma_commit();
}

template <int WBITS>
__global__ __launch_bounds__(512) void lowbit_wgmma_ptx(const int8_t * __restrict__ Aq,
                                                        const float * __restrict__ dA,
                                                        const unsigned * __restrict__ Wbits,
                                                        const float * __restrict__ Wd,
                                                        float * __restrict__ C,
                                                        int M,
                                                        int N,
                                                        int K) {
    extern __shared__ __align__(1024) int8_t smem[];
    int8_t *                                 sA[2] = { smem, smem + SMEM_TILE };
    int8_t *                                 sB[2] = { smem + 2 * SMEM_TILE, smem + 3 * SMEM_TILE };
    float *                                  sDa[2];
    float *                                  sDw[2];
    {
        float * p = reinterpret_cast<float *>(smem + 4 * SMEM_TILE);
        sDa[0]    = p;
        sDa[1]    = p + bM;
        sDw[0]    = p + 2 * bM;
        sDw[1]    = p + 2 * bM + bN;
    }
    const int mblk = blockIdx.x, nblk = blockIdx.y;
    const int wg   = threadIdx.x / 128;
    const int warp = (threadIdx.x % 128) / 32, lane = threadIdx.x % 32;
    const int wgm = wg / 2, wgn = wg % 2;

    int32_t acc_i32[32];
    float   acc_f32[32] = { 0.f };

    const int nchunks = K / bK;
    load_stage_ptx<WBITS>(sA[0], sB[0], sDa[0], sDw[0], Aq, dA, Wbits, Wd, mblk, nblk, 0, K);
    for (int kc = 0; kc < nchunks; ++kc) {
        int cur = kc & 1, nxt = cur ^ 1;
        __pipeline_wait_prior(0);  // stage `cur` cp.async complete before it is published
        fence_proxy_async();       // publish generic-proxy unpack stores to the wgmma proxy
        __syncthreads();
        mma_stage(acc_i32, sA[cur] + wgm * (64 * bK), sB[cur] + wgn * (64 * bK));
        if (kc + 1 < nchunks) {
            load_stage_ptx<WBITS>(sA[nxt], sB[nxt], sDa[nxt], sDw[nxt], Aq, dA, Wbits, Wd, mblk, nblk, kc + 1, K);
        }
        wgmma_wait0();
#    pragma unroll
        for (int e = 0; e < 32; ++e) {
            int ml, nl;
            acc_coord(e, warp, lane, ml, nl);
            acc_f32[e] += float(acc_i32[e]) * (sDa[cur][wgm * 64 + ml] * sDw[cur][wgn * 64 + nl]);
        }
    }
#    pragma unroll
    for (int e = 0; e < 32; ++e) {
        int ml, nl;
        acc_coord(e, warp, lane, ml, nl);
        int m                 = mblk * bM + wgm * 64 + ml;
        int n                 = nblk * bN + wgn * 64 + nl;
        C[(size_t) m * N + n] = acc_f32[e];
    }
}

// stream-K variant: persistent work-centric loop; scaled fp32 partials atomicAdd'd into
// pre-zeroed C (same scheme as the CuTe path)
template <int WBITS>
__global__ __launch_bounds__(512) void lowbit_wgmma_ptx_sk(const int8_t * __restrict__ Aq,
                                                           const float * __restrict__ dA,
                                                           const unsigned * __restrict__ Wbits,
                                                           const float * __restrict__ Wd,
                                                           float * __restrict__ C,
                                                           int  M,
                                                           int  N,
                                                           int  K,
                                                           int  ntn,
                                                           long total,
                                                           int  ncta) {
    extern __shared__ __align__(1024) int8_t smem[];
    int8_t *                                 sA[2] = { smem, smem + SMEM_TILE };
    int8_t *                                 sB[2] = { smem + 2 * SMEM_TILE, smem + 3 * SMEM_TILE };
    float *                                  sDa[2];
    float *                                  sDw[2];
    {
        float * p = reinterpret_cast<float *>(smem + 4 * SMEM_TILE);
        sDa[0]    = p;
        sDa[1]    = p + bM;
        sDw[0]    = p + 2 * bM;
        sDw[1]    = p + 2 * bM + bN;
    }
    int       mblk = 0, nblk = 0;
    const int wg   = threadIdx.x / 128;
    const int warp = (threadIdx.x % 128) / 32, lane = threadIdx.x % 32;
    const int wgm = wg / 2, wgn = wg % 2;

    int32_t acc_i32[32];
    float   acc_f32[32] = { 0.f };

    const int nchunks = K / 128;
    auto      flush   = [&](bool owned) {
#    pragma unroll
        for (int e = 0; e < 32; ++e) {
            int ml, nl;
            acc_coord(e, warp, lane, ml, nl);
            int m = mblk * bM + wgm * 64 + ml;
            int n = nblk * bN + wgn * 64 + nl;
            if (owned) {
                C[(size_t) m * N + n] = acc_f32[e];             // whole tile in this CTA's span
            } else {
                atomicAdd(&C[(size_t) m * N + n], acc_f32[e]);  // split tile (C pre-zeroed)
            }
            acc_f32[e] = 0.f;
        }
    };
    auto tile_owned = [&](int tile, long lo, long hi) {
        long first = (long) tile * nchunks, last = first + nchunks - 1;
        return first >= lo && last < hi;
    };
    const int cta = blockIdx.x;
    long      u0 = (long) cta * total / ncta, u1 = (long) (cta + 1) * total / ncta;
    int       cur_tile = -1, buf = 0;
    for (long u = u0; u < u1; ++u) {
        int tile = (int) (u / nchunks), kc = (int) (u % nchunks);
        if (tile != cur_tile) {
            if (cur_tile >= 0) {
                flush(tile_owned(cur_tile, u0, u1));
            }
            cur_tile = tile;
            mblk     = tile / ntn;
            nblk     = tile % ntn;
            buf      = 0;
            load_stage_ptx<WBITS>(sA[buf], sB[buf], sDa[buf], sDw[buf], Aq, dA, Wbits, Wd, mblk, nblk, kc, K);
        }
        __pipeline_wait_prior(0);  // stage `buf` cp.async complete before it is published
        fence_proxy_async();
        __syncthreads();
        mma_stage(acc_i32, sA[buf] + wgm * (64 * bK), sB[buf] + wgn * (64 * bK));
        bool next_same = (u + 1 < u1) && ((u + 1) / nchunks == tile);
        if (next_same) {
            load_stage_ptx<WBITS>(sA[buf ^ 1], sB[buf ^ 1], sDa[buf ^ 1], sDw[buf ^ 1], Aq, dA, Wbits, Wd, mblk, nblk,
                                  kc + 1, K);
        }
        wgmma_wait0();
#    pragma unroll
        for (int e = 0; e < 32; ++e) {
            int ml, nl;
            acc_coord(e, warp, lane, ml, nl);
            acc_f32[e] += float(acc_i32[e]) * (sDa[buf][wgm * 64 + ml] * sDw[buf][wgn * 64 + nl]);
        }
        if (next_same) {
            buf ^= 1;
        }
        __syncthreads();
    }
    if (cur_tile >= 0) {
        flush(tile_owned(cur_tile, u0, u1));
    }
}

}  // namespace hopper_q1_ptx
#endif  // GGML_USE_HOPPER_Q1_PTX

// runtime-selected twin of ggml_cuda_mul_mat_q1_hopper (same contract: false = fall through)
bool ggml_cuda_mul_mat_q1_hopper_ptx(ggml_backend_cuda_context & ctx,
                                     const ggml_tensor *         src0,
                                     const ggml_tensor *         src1,
                                     ggml_tensor *               dst) {
#if defined(GGML_USE_HOPPER_Q1_PTX)
    static const bool enabled = getenv("GGML_HOPPER_Q1") != nullptr;
    if (!enabled) {
        return false;
    }
    const int     cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    const bool    is_q1 = src0->type == GGML_TYPE_Q1_0;
    const bool    is_q2 = src0->type == GGML_TYPE_Q2_0;
    if (cc < 900 ||  // 900 = Hopper; no GGML_CUDA_CC_HOPPER macro in this tree
        (!is_q1 && !is_q2) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src1->ne[2] * src1->ne[3] != 1 || src0->ne[2] * src0->ne[3] != 1 || (M % 128) || (N % 128) || (K % 128) ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    cudaStream_t                 stream = ctx.stream();
    hopper_q1_ptx::DenseW        wq = hopper_q1_ptx::get_dense_w(src0->data, (long) N, (long) K, is_q2 ? 2 : 1, stream);
    ggml_cuda_pool_alloc<int8_t> act_q(ctx.pool(), (size_t) M * K);
    ggml_cuda_pool_alloc<float>  act_d(ctx.pool(), (size_t) M * (K / 128));
    {
        const int ngroups = (int) (M * (K / 128));
        hopper_q1_ptx::quant_act_per128<<<(ngroups + 7) / 8, 256, 0, stream>>>((const float *) src1->data, act_q.get(),
                                                                               act_d.get(), (int) M, (int) K);
    }
    auto *      kern_fixed = is_q2 ? hopper_q1_ptx::lowbit_wgmma_ptx<2> : hopper_q1_ptx::lowbit_wgmma_ptx<1>;
    auto *      kern_sk    = is_q2 ? hopper_q1_ptx::lowbit_wgmma_ptx_sk<2> : hopper_q1_ptx::lowbit_wgmma_ptx_sk<1>;
    static bool attr_set   = false;
    if (!attr_set) {
        cudaFuncSetAttribute(hopper_q1_ptx::lowbit_wgmma_ptx<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             hopper_q1_ptx::SMEM_BYTES);
        cudaFuncSetAttribute(hopper_q1_ptx::lowbit_wgmma_ptx<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             hopper_q1_ptx::SMEM_BYTES);
        cudaFuncSetAttribute(hopper_q1_ptx::lowbit_wgmma_ptx_sk<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             hopper_q1_ptx::SMEM_BYTES);
        cudaFuncSetAttribute(hopper_q1_ptx::lowbit_wgmma_ptx_sk<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             hopper_q1_ptx::SMEM_BYTES);
        attr_set = true;
    }
    const int  ntm = (int) (M / hopper_q1_ptx::bM), ntn = (int) (N / hopper_q1_ptx::bN);
    const int  ntiles = ntm * ntn;
    static int NSM    = 0;
    if (NSM == 0) {
        cudaDeviceGetAttribute(&NSM, cudaDevAttrMultiProcessorCount, ctx.device);
    }
    if (ntiles < 8 * NSM) {
        // starved grid -> stream-K (persistent; fp32-additive atomic flush into zeroed C)
        const long total = (long) ntiles * (K / 128);
        cudaMemsetAsync(dst->data, 0, (size_t) M * N * sizeof(float), stream);
        kern_sk<<<NSM, 512, hopper_q1_ptx::SMEM_BYTES, stream>>>(
            act_q.get(), act_d.get(), wq.bits, wq.dw, (float *) dst->data, (int) M, (int) N, (int) K, ntn, total, NSM);
    } else {
        dim3 grid(ntm, ntn);
        kern_fixed<<<grid, 512, hopper_q1_ptx::SMEM_BYTES, stream>>>(act_q.get(), act_d.get(), wq.bits, wq.dw,
                                                                     (float *) dst->data, (int) M, (int) N, (int) K);
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    return false;
#endif
}
