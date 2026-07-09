import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
import torch
from test_common import run_perftest, verify_output
from utils import pertoken_quant


def compile_fp8_gemm_4wave_256x256x128(
    *,
    M: int,
    N: int,
    K: int,
):
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K

    assert N % BLOCK_N == 0
    assert M % BLOCK_M == 0
    assert K % BLOCK_K == 0

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
    ):
        # === Type declarations ===
        MfmaAccumType_t = Vec.make_type(4, fx.Float32)
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)

        lds_alloc = fx.SharedAllocator()
        A_lds = [
            # These are FlyDSL `Pointer` objects
            [
                lds_alloc.allocate(fx.Array[fx.Float8E4M3FN, 128 * 128, 16]).peek().ptr
                for _ in range_constexpr(2)
            ]
            for _ in range_constexpr(2)
        ]

        B_lds = [
            [
                lds_alloc.allocate(fx.Array[fx.Float8E4M3FN, 128 * 128, 16]).peek().ptr
                for _ in range_constexpr(2)
            ]
            for _ in range_constexpr(2)
        ]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        tile_i = fx.block_idx.x // N_BLOCKS
        tile_j = fx.block_idx.x % N_BLOCKS
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A128_gl_offset = (tile_i * BLOCK_M + 128) * K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B128_gl_offset = (tile_j * BLOCK_N + 128) * K

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B_T)
        C_rsrc = buffer_ops.create_buffer_resource(C)

        def _swizzle_128(row, col):
            offset = row * 128 + col
            swizzle = ((offset % (16 * 128)) >> 8) << 4
            swizzled_offset = offset ^ swizzle
            return swizzled_offset // 128, swizzled_offset % 128

        def _compute_global_swizzle():
            offsets = []
            for round in range_constexpr(4):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                a, b = _swizzle_128(row, col)
                offsets.append(a * K + b)
            return offsets

        def _compute_lds_swizzle(wave_idx):
            lds_swz = []
            for row_offset in range_constexpr(4):
                row = wave_idx * 64 + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    swz_row, swz_col = _swizzle_128(row, col)
                    swz.append(swz_row * 128 + swz_col)
                lds_swz.append(swz)
            return lds_swz

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets):
            lds_base = fx.Int32(fx.ptrtoint(lds_dst))
            for step in range_constexpr(4):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_base + fx.Int32(wave_id * 1024 + step * 4096), address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src,
                    lds_ptr,
                    fx.Int32(16),
                    fx.Int32(gl_offsets[step]),  # voffset
                    fx.Int32(k_offset),  # soffset
                    fx.Int32(0),
                    fx.Int32(0),
                )

        def _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, step):
            lds_base = fx.Int32(fx.ptrtoint(lds_dst))
            lds_ptr = buffer_ops.create_llvm_ptr(
                lds_base + fx.Int32(wave_id * 1024 + step * 4096), address_space=3
            )
            rocdl.raw_ptr_buffer_load_lds(
                gl_src,
                lds_ptr,
                fx.Int32(16),
                fx.Int32(gl_offsets[step]),  # voffset
                fx.Int32(k_offset),  # soffset
                fx.Int32(0),
                fx.Int32(0),
            )

        def _pack_i32x42_i32x8(lo, hi):
            # Pack 2 i32x4 as i32x8
            return lo.shuffle(hi, list(range(8)))

        # Shim object: `ptr_load` will call `result_type.ir_type`
        # because it expects a FlyDSL object, not an MLIR value.
        # This will probably be fixed in some future version.
        class I32x4:
            ir_type = Vec.make_type(4, fx.Int32)

        def _lds_load_i32x4(lds_ptr, elem_offset):
            i32_ptr = fx.recast_iter(fx.Uint8, lds_ptr + elem_offset)
            return fx.ptr_load(i32_ptr, result_type=I32x4)

        def _load_rt(lds_src, wave_idx):
            # Load a 64x128 fragment of A/B from LDS to registers
            # Each 16x128 fragment requires 2 i32x4 (2 ds_read_b128)
            frag = []
            for i in range_constexpr(4):
                row = wave_idx * 64 + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    halves.append(_lds_load_i32x4(lds_src, row_swz * 128 + col_swz))
                frag.append(_pack_i32x42_i32x8(halves[0], halves[1]))  # i32x8
            return frag

        def _load_one_rt(lds_src, lds_swz, row, k):
            # Load half of a 16x128 tile from LDS to registers
            return _lds_load_i32x4(lds_src, lds_swz[row][k])  # i32x4

        def _store_rt(c_frag, base_row, base_col):
            for ti in range_constexpr(4):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(4):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_bf16 = Vec(c_frag[ti][tj]).to(fx.BFloat16)
                    for i in range_constexpr(4):
                        buffer_ops.buffer_store(
                            vec_bf16[i], C_rsrc, fx.Int32((row + i) * N + col)
                        )

        def _do_mfma(a, b, c):
            return _llvm.inline_asm(
                MfmaAccumType_t,
                [arith._to_raw(a), arith._to_raw(b), arith._to_raw(c)],
                "v_mfma_f32_16x16x128_f8f6f4 $0, $1, $2, $0",
                "=a,v,v,0",
                has_side_effects=True,
            )

        def _mfma_ABt(a, b, c, m, n):
            c[m][n] = _do_mfma(a[m], b[n], c[m][n])
            return c

        def _mfma_ABt_all(a, b, c):
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    c[i][j] = _do_mfma(a[i], b[j], c[i][j])
            return c

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\n\ts_barrier",
                constraints="",
                has_side_effects=True,
            )

        def _interleaved_cluster(
            lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, a, b, c
        ):
            # Compute a 64x64 output tile using 4x4 MFMA instructions
            # returns the updated accumulator and the next fragment loaded from lds_src
            rt_dst = []

            c = _mfma_ABt(a, b, c, 0, 0)
            c = _mfma_ABt(a, b, c, 0, 1)

            lds_swz = _compute_lds_swizzle(wave_idx)
            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 0)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 0, 0)

            c = _mfma_ABt(a, b, c, 0, 2)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 0, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt(a, b, c, 0, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 1)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 1, 0)

            c = _mfma_ABt(a, b, c, 1, 0)
            c = _mfma_ABt(a, b, c, 1, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 1, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt(a, b, c, 1, 2)
            c = _mfma_ABt(a, b, c, 1, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 2)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 2, 0)

            c = _mfma_ABt(a, b, c, 2, 0)
            c = _mfma_ABt(a, b, c, 2, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 2, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt(a, b, c, 2, 2)
            c = _mfma_ABt(a, b, c, 2, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 3)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 3, 0)

            c = _mfma_ABt(a, b, c, 3, 0)
            c = _mfma_ABt(a, b, c, 3, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 3, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt(a, b, c, 3, 2)
            c = _mfma_ABt(a, b, c, 3, 3)

            return c, rt_dst

        # Each wave handles 2x2 64x64 sub-tiles of the output
        c00_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]
        c01_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]
        c10_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]
        c11_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]

        global_offsets = _compute_global_swizzle()

        # Prologue: pre-load A/B cur
        _load_lds(A_rsrc, A_lds[0][0], A0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[0][0], B0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[0][1], B128_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, A_lds[0][1], A128_gl_offset + 0 * BLOCK_K, global_offsets)

        # Issue load for next tile
        _load_lds(A_rsrc, A_lds[1][0], A0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[1][0], B0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[1][1], B128_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, A_lds[1][1], A128_gl_offset + 1 * BLOCK_K, global_offsets)

        _wait_barrier(28)

        a0_frag = _load_rt(A_lds[0][0], wave_i)

        _wait_barrier(24)

        b0_frag = _load_rt(B_lds[0][0], wave_j)

        cur, next = 0, 1
        for k in range_constexpr(K_ITERS - 2):
            _wait_barrier(16)

            c00_frag, b1_frag = _interleaved_cluster(
                A_lds[cur][0],
                A_rsrc,
                A0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_j,
                B_lds[cur][1],
                a0_frag,
                b0_frag,
                c00_frag,
            )

            c01_frag, a1_frag = _interleaved_cluster(
                B_lds[cur][0],
                B_rsrc,
                B0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_i,
                A_lds[cur][1],
                a0_frag,
                b1_frag,
                c01_frag,
            )

            _wait_barrier(16)

            c10_frag, a0_frag = _interleaved_cluster(
                B_lds[cur][1],
                B_rsrc,
                B128_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_i,
                A_lds[next][0],
                a1_frag,
                b0_frag,
                c10_frag,
            )

            c11_frag, b0_frag = _interleaved_cluster(
                A_lds[cur][1],
                A_rsrc,
                A128_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_j,
                B_lds[next][0],
                a1_frag,
                b1_frag,
                c11_frag,
            )

            # Swap cur and next
            cur ^= 1
            next ^= 1

        # step k = k_iters - 2
        _wait_barrier(16)

        b1_frag = _load_rt(B_lds[cur][1], wave_j)

        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)

        a1_frag = _load_rt(A_lds[cur][1], wave_i)

        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)

        _wait_barrier(8)

        a0_frag = _load_rt(A_lds[next][0], wave_i)

        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)

        b0_frag = _load_rt(B_lds[next][0], wave_j)

        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

        # Swap cur and next
        cur ^= 1
        next ^= 1

        # step k = k_iters - 1
        base_row = tile_i * BLOCK_M + wave_i * 64
        base_col = tile_j * BLOCK_N + wave_j * 64

        _wait_barrier(0)

        b1_frag = _load_rt(B_lds[cur][1], wave_j)
        a1_frag = _load_rt(A_lds[cur][1], wave_i)

        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        _store_rt(c00_frag, base_row + 0, base_col + 0)

        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        _store_rt(c01_frag, base_row + 0, base_col + 128)

        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        _store_rt(c10_frag, base_row + 128, base_col + 0)

        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        _store_rt(c11_frag, base_row + 128, base_col + 128)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        grid_x = (M * N) // (256 * 256)
        kernel_gemm(
            A,
            B_T,
            C,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm


def _run_torch(a, b, dtype=torch.float32):
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)


def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16


def check_fp8_gemm(
    M: int,
    N: int,
    K: int,
    num_warmups: int = 2,
    num_iters: int = 10,
) -> tuple[float, float]:
    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    c_out_raw = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()

    launch_gemm_fn = compile_fp8_gemm_4wave_256x256x128(M=M, N=N, K=K)
    stream = torch.cuda.current_stream()

    def _gemm_args(c, a, b):
        return (
            _as_i8(a).contiguous().view(-1),
            _as_i8(b).contiguous().view(-1),
            c.contiguous().view(-1),
            stream,
        )

    compiled_gemm = flyc.compile(launch_gemm_fn, *_gemm_args(c_out_raw, a_q, b_q))

    def _launch(c, a, b):
        compiled_gemm(*_gemm_args(c, a, b))

    num_iters = max(2, int(num_iters))

    _, us = run_perftest(
        _launch,
        c_out_raw,
        a_q,
        b_q,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    c_ref = _run_torch(a_q, b_q)
    c_out_f32 = c_out_raw.to(torch.float32)
    assert verify_output(c_out_f32, c_ref, rtol=0.1, atol=0.1)

    flops = 2 * M * N * K
    return flops / (us / 1e6) / 1e12, us


if __name__ == "__main__":
    for s in [4, 8, 12, 16]:
        m = n = k = s * 1024
        print(f"Benchmarking {m}...")
        tflops, us = check_fp8_gemm(m, n, k, num_iters=1000, num_warmups=1000)

        print(f"FP8 GEMM {m}x{n}x{k} FlyDSL: {tflops:.1f}TFLOPS ({us}us)")
