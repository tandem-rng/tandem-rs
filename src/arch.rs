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

/// The eight lanes of a group in 256-bit registers, one register per state word, selected at
/// run time. The portable code keeps the same state in two 128-bit halves.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
mod avx2 {
    use core::arch::x86_64::*;

    use wide::u32x4;

    use crate::{Rows, Tandem};

    const CLOCK_WEYL: i32 = crate::CLOCK_WEYL as i32;

    #[derive(Clone, Copy)]
    pub(super) struct Avx2Lanes {
        o: [__m256i; 4],
        h: [__m256i; 4],
    }

    /// Whether the CPU runs the 256-bit path. Setting `TANDEM_NO_AVX2` turns it off, which
    /// keeps the 128-bit path under test on a CPU that has AVX2.
    pub(crate) fn available() -> bool {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            std::env::var_os("TANDEM_NO_AVX2").is_none() && std::is_x86_feature_detected!("avx2")
        })
    }

    /// Whether the normal loops can use fused multiply-add instructions: the same switch as
    /// `available`, because every x86_64 CPU with AVX2 has FMA3 in practice and the check says
    /// so.
    pub(crate) fn fma_available() -> bool {
        available() && std::is_x86_feature_detected!("fma")
    }

    /// The normal and exponential loops compiled with FMA, so `mul_add` is one instruction and the loop
    /// vectorises. The bits are the same as without.
    pub(crate) fn block_f64_fma(u: &[f64], z: &mut [f64]) {
        assert!(fma_available(), "AVX2 and FMA are required");
        // SAFETY: the check above is the requirement of the target features.
        unsafe { block_f64_avx2_fma(u, z) }
    }

    pub(crate) fn block_f32_fma(u: &[f32], z: &mut [f32]) {
        assert!(fma_available(), "AVX2 and FMA are required");
        // SAFETY: the check above is the requirement of the target features.
        unsafe { block_f32_avx2_fma(u, z) }
    }

    pub(crate) fn exponential_block_f64_fma(z: &mut [f64]) {
        assert!(fma_available(), "AVX2 and FMA are required");
        // SAFETY: the check above is the requirement of the target features.
        unsafe { exponential_block_f64_avx2_fma(z) }
    }

    pub(crate) fn exponential_block_f32_fma(z: &mut [f32]) {
        assert!(fma_available(), "AVX2 and FMA are required");
        // SAFETY: the check above is the requirement of the target features.
        unsafe { exponential_block_f32_avx2_fma(z) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn exponential_block_f64_avx2_fma(z: &mut [f64]) {
        crate::boxmuller::exponential_block_f64_body(z)
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn exponential_block_f32_avx2_fma(z: &mut [f32]) {
        crate::boxmuller::exponential_block_f32_body(z)
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn block_f64_avx2_fma(u: &[f64], z: &mut [f64]) {
        crate::boxmuller::block_f64_body(u, z)
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn block_f32_avx2_fma(u: &[f32], z: &mut [f32]) {
        crate::boxmuller::block_f32_body(u, z)
    }

    /// The loop of `run_rows` on 256-bit registers.
    ///
    /// # Panics
    ///
    /// Without AVX2, which `available` reports.
    pub(crate) fn run_rows_avx2<S: FnMut(&[u32x4; 8])>(
        rng: &mut Tandem,
        row: u64,
        nrows: u64,
        sink: Option<S>,
    ) {
        assert!(available(), "AVX2 is required");
        // SAFETY: AVX2 was just checked.
        unsafe { run_avx2(rng, row, nrows, sink) }
    }

    // Every helper of `Avx2Lanes` is `inline(always)`, so it lands in this function, which
    // has the feature.
    #[target_feature(enable = "avx2")]
    unsafe fn run_avx2<S: FnMut(&[u32x4; 8])>(
        rng: &mut Tandem,
        row: u64,
        nrows: u64,
        sink: Option<S>,
    ) {
        rng.run_rows_with::<Avx2Lanes, S>(row, nrows, sink)
    }

    // The methods are safe to call only from `run_rows_avx2`, the one place that builds an
    // `Avx2Lanes`, and so always run with AVX2 enabled.
    impl Avx2Lanes {
        /// Low and high words of the eight 32x32 to 64-bit products: `pmuludq` takes the even
        /// lanes, so the odd ones go through a shift.
        #[inline(always)]
        fn mul_wide(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
            // SAFETY: AVX2 is enabled, see above.
            unsafe {
                let even = _mm256_mul_epu32(a, b);
                let odd = _mm256_mul_epu32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(b));
                (
                    _mm256_blend_epi32::<0b1010_1010>(even, _mm256_slli_epi64::<32>(odd)),
                    _mm256_blend_epi32::<0b1010_1010>(_mm256_srli_epi64::<32>(even), odd),
                )
            }
        }

        #[inline(always)]
        fn rotl<const L: i32, const R: i32>(x: __m256i) -> __m256i {
            // SAFETY: AVX2 is enabled, see above.
            unsafe { _mm256_or_si256(_mm256_slli_epi32::<L>(x), _mm256_srli_epi32::<R>(x)) }
        }
    }

    impl Rows for Avx2Lanes {
        #[inline(always)]
        fn load(o: &[[u32; 8]; 4], h: &[[u32; 8]; 4]) -> Self {
            // SAFETY: AVX2 is enabled, and the arrays hold 32 bytes each.
            unsafe {
                let ld = |w: &[u32; 8]| _mm256_loadu_si256(w.as_ptr().cast());
                Avx2Lanes {
                    o: o.each_ref().map(ld),
                    h: h.each_ref().map(ld),
                }
            }
        }

        #[inline(always)]
        fn save(&self, o: &mut [[u32; 8]; 4], h: &mut [[u32; 8]; 4]) {
            // SAFETY: AVX2 is enabled, and the arrays hold 32 bytes each.
            unsafe {
                for w in 0..4 {
                    _mm256_storeu_si256(o[w].as_mut_ptr().cast(), self.o[w]);
                    _mm256_storeu_si256(h[w].as_mut_ptr().cast(), self.h[w]);
                }
            }
        }

        #[inline(always)]
        fn step(&mut self) {
            let (o, h) = (&mut self.o, &mut self.h);
            // SAFETY: AVX2 is enabled, see above.
            unsafe {
                let one = _mm256_set1_epi32(1);
                let (lo0, hi0) = Self::mul_wide(o[0], _mm256_or_si256(h[0], one));
                let (lo1, hi1) = Self::mul_wide(o[2], _mm256_or_si256(h[1], one));
                let n0 = _mm256_xor_si256(_mm256_xor_si256(o[1], hi1), lo1);
                let n1 = _mm256_xor_si256(Self::rotl::<16, 16>(lo1), h[2]);
                let n2 = _mm256_xor_si256(_mm256_xor_si256(o[3], hi0), lo0);
                let n3 = _mm256_xor_si256(Self::rotl::<16, 16>(lo0), h[3]);
                h[0] = _mm256_xor_si256(h[0], Self::rotl::<7, 25>(h[1]));
                h[1] = _mm256_xor_si256(h[1], Self::rotl::<13, 19>(h[2]));
                h[2] = _mm256_xor_si256(h[2], Self::rotl::<22, 10>(h[3]));
                h[3] = _mm256_xor_si256(h[3], Self::rotl::<3, 29>(h[0]));
                h[0] = _mm256_xor_si256(_mm256_add_epi32(h[0], _mm256_set1_epi32(CLOCK_WEYL)), n0);
                *o = [n0, n1, n2, n3];
            }
        }

        #[inline(always)]
        fn seed(&mut self, key: &[u32; 4], g: u64) {
            let c0 = 8 * g;
            let lo = c0 as u32;
            // SAFETY: AVX2 is enabled, see above.
            unsafe {
                let mut q = Avx2Lanes {
                    o: [
                        _mm256_add_epi32(
                            _mm256_set1_epi32(lo as i32),
                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                        ),
                        _mm256_set1_epi32((c0 >> 32) as i32),
                        _mm256_set1_epi32(crate::DOMAIN_STREAM as i32),
                        _mm256_set1_epi32(crate::AUX_STREAM as i32),
                    ],
                    h: key.map(|w| _mm256_set1_epi32(w as i32)),
                };
                for rc in crate::RC {
                    q.step();
                    q.o[0] = _mm256_xor_si256(q.o[0], _mm256_set1_epi32(rc as i32));
                    q = Avx2Lanes { o: q.h, h: q.o };
                }
                *self = q;
            }
        }

        /// The row's eight blocks: the 4x8 words transposed within each 128-bit half, so the
        /// low halves hold blocks 0 to 3 and the high halves blocks 4 to 7.
        #[inline(always)]
        fn blocks(&self) -> [u32x4; 8] {
            // SAFETY: AVX2 is enabled, see above.
            unsafe {
                let o = &self.o;
                let t0 = _mm256_unpacklo_epi32(o[0], o[1]);
                let t1 = _mm256_unpackhi_epi32(o[0], o[1]);
                let t2 = _mm256_unpacklo_epi32(o[2], o[3]);
                let t3 = _mm256_unpackhi_epi32(o[2], o[3]);
                let u = [
                    _mm256_unpacklo_epi64(t0, t2),
                    _mm256_unpackhi_epi64(t0, t2),
                    _mm256_unpacklo_epi64(t1, t3),
                    _mm256_unpackhi_epi64(t1, t3),
                ];
                let half = |v: __m256i, hi: bool| -> u32x4 {
                    if hi {
                        _mm256_extracti128_si256::<1>(v).into()
                    } else {
                        _mm256_castsi256_si128(v).into()
                    }
                };
                [
                    half(u[0], false),
                    half(u[1], false),
                    half(u[2], false),
                    half(u[3], false),
                    half(u[0], true),
                    half(u[1], true),
                    half(u[2], true),
                    half(u[3], true),
                ]
            }
        }
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub(crate) use avx2::{
    available as avx2_available, block_f32_fma, block_f64_fma, exponential_block_f32_fma,
    exponential_block_f64_fma, fma_available, run_rows_avx2,
};
