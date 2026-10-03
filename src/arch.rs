//! Target intrinsics behind the `simd-intrinsics` feature: the widening multiply and the float
//! row stores, spelled as the instructions that `wide` and LLVM do not pick on their own.
//! Every function returns what the portable code returns, bit for bit.
//!
//! The intrinsics need `unsafe` even when the target feature is part of the build
//! configuration, so this is the one module that may use it. Each block depends only on the
//! `cfg` of its module, which guarantees the target feature at compile time.

#![allow(unsafe_code)]

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
mod imp {
    use core::arch::aarch64::*;

    use wide::{f32x4, f64x2, u32x4};

    /// Low and high words of the four 32x32 to 64-bit products: `umull`, `umull2` and two
    /// unzips instead of `mul` beside the high-half `umull` pair.
    #[inline(always)]
    pub(crate) fn mul_wide(a: u32x4, b: u32x4) -> (u32x4, u32x4) {
        let (a, b): (uint32x4_t, uint32x4_t) = (a.into(), b.into());
        // SAFETY: the module's cfg guarantees `neon`, the only requirement of these intrinsics.
        unsafe {
            let p0 = vreinterpretq_u32_u64(vmull_u32(vget_low_u32(a), vget_low_u32(b)));
            let p1 = vreinterpretq_u32_u64(vmull_high_u32(a, b));
            (vuzp1q_u32(p0, p1).into(), vuzp2q_u32(p0, p1).into())
        }
    }

    /// `(raw >> 8) * 2^-24` per lane: `ucvtf` with 24 fraction bits scales for free.
    #[inline(always)]
    pub(crate) fn store_f32(blocks: &[u32x4; 8], dst: &mut [f32; 32]) {
        for (block, out) in blocks.iter().zip(dst.as_chunks_mut::<4>().0) {
            let raw: uint32x4_t = (*block).into();
            // SAFETY: the module's cfg guarantees `neon`, the only requirement of these intrinsics.
            let f = unsafe { vcvtq_n_f32_u32::<24>(vshrq_n_u32::<8>(raw)) };
            *out = bytemuck::cast(f32x4::from(f));
        }
    }

    /// `(raw >> 11) * 2^-53` per lane, with the block's words paired little-endian.
    #[inline(always)]
    pub(crate) fn store_f64(blocks: &[u32x4; 8], dst: &mut [f64; 16]) {
        for (block, out) in blocks.iter().zip(dst.as_chunks_mut::<2>().0) {
            let raw: uint32x4_t = (*block).into();
            // SAFETY: the module's cfg guarantees `neon`, the only requirement of these intrinsics.
            let f = unsafe { vcvtq_n_f64_u64::<53>(vshrq_n_u64::<11>(vreinterpretq_u64_u32(raw))) };
            *out = bytemuck::cast(f64x2::from(f));
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod imp {
    use core::arch::x86_64::*;

    use wide::{f32x4, f64x2, u32x4};

    /// Low and high words of the four 32x32 to 64-bit products from two `pmuludq`, which
    /// multiply lanes 0 and 2. The portable code spends four.
    #[inline(always)]
    pub(crate) fn mul_wide(a: u32x4, b: u32x4) -> (u32x4, u32x4) {
        let (a, b): (__m128i, __m128i) = (a.into(), b.into());
        // SAFETY: the module's cfg guarantees `sse2`, the only requirement of these intrinsics.
        unsafe {
            let p02 = _mm_mul_epu32(a, b);
            let p13 = _mm_mul_epu32(_mm_srli_epi64::<32>(a), _mm_srli_epi64::<32>(b));
            // Words 0 and 2 of each product vector are the low halves, 1 and 3 the high.
            let lo = _mm_unpacklo_epi32(
                _mm_shuffle_epi32::<0b00_00_10_00>(p02),
                _mm_shuffle_epi32::<0b00_00_10_00>(p13),
            );
            let hi = _mm_unpacklo_epi32(
                _mm_shuffle_epi32::<0b00_00_11_01>(p02),
                _mm_shuffle_epi32::<0b00_00_11_01>(p13),
            );
            (lo.into(), hi.into())
        }
    }

    /// `(raw >> 8) * 2^-24` per lane. The shifted word fits in an `i32`, so the signed
    /// convert is exact.
    #[inline(always)]
    pub(crate) fn store_f32(blocks: &[u32x4; 8], dst: &mut [f32; 32]) {
        for (block, out) in blocks.iter().zip(dst.as_chunks_mut::<4>().0) {
            let raw: __m128i = (*block).into();
            // SAFETY: the module's cfg guarantees `sse2`, the only requirement of these intrinsics.
            let f = unsafe {
                _mm_mul_ps(
                    _mm_cvtepi32_ps(_mm_srli_epi32::<8>(raw)),
                    _mm_set1_ps(1.0 / (1u32 << 24) as f32),
                )
            };
            *out = bytemuck::cast(f32x4::from(f));
        }
    }

    /// `(raw >> 11) * 2^-53` per lane without a 64-bit convert, which SSE2 lacks. With
    /// `hi = raw >> 32` and `lo = (raw >> 11) mod 2^21`, the doubles with exponents 2^20 and
    /// 2^-1 and those bits as significand are `2^20 + hi * 2^-32` and `2^-1 + lo * 2^-53`.
    /// Subtracting `2^20 + 2^-1` from the first and adding the second is exact at each step,
    /// since every result has at most 53 significant bits.
    #[inline(always)]
    pub(crate) fn store_f64(blocks: &[u32x4; 8], dst: &mut [f64; 16]) {
        const HI_EXP: i64 = 0x4130_0000_0000_0000; // 2^20
        const LO_EXP: i64 = 0x3fe0_0000_0000_0000; // 2^-1
        for (block, out) in blocks.iter().zip(dst.as_chunks_mut::<2>().0) {
            let raw: __m128i = (*block).into();
            // SAFETY: the module's cfg guarantees `sse2`, the only requirement of these intrinsics.
            let f = unsafe {
                let hi = _mm_or_si128(_mm_srli_epi64::<32>(raw), _mm_set1_epi64x(HI_EXP));
                let lo = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi64::<11>(raw), _mm_set1_epi64x(0x1f_ffff)),
                    _mm_set1_epi64x(LO_EXP),
                );
                _mm_add_pd(
                    _mm_sub_pd(_mm_castsi128_pd(hi), _mm_set1_pd(1048576.5)),
                    _mm_castsi128_pd(lo),
                )
            };
            *out = bytemuck::cast(f64x2::from(f));
        }
    }
}

#[cfg(not(any(
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ),
    all(target_arch = "x86_64", target_feature = "sse2"),
)))]
mod imp {
    use wide::u32x4;

    #[inline(always)]
    pub(crate) fn mul_wide(a: u32x4, b: u32x4) -> (u32x4, u32x4) {
        (a * b, a.mul_keep_high(b))
    }

    #[inline(always)]
    pub(crate) fn store_f32(blocks: &[u32x4; 8], dst: &mut [f32; 32]) {
        crate::store_le(blocks, dst)
    }

    #[inline(always)]
    pub(crate) fn store_f64(blocks: &[u32x4; 8], dst: &mut [f64; 16]) {
        crate::store_le(blocks, dst)
    }
}

pub(crate) use imp::*;
