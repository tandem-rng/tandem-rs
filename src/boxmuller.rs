//! The `f32` Box-Muller normals on whole blocks of pairs, the exponentials, and the reference
//! logarithm of Appendix A, in plain Rust that LLVM vectorises. The shape and the coefficients
//! are tandem-c's, so the ports compute the same arithmetic.
//!
//! The exponentials `-ln(1 - a)` and the `f64` ziggurat share the logarithm. The exponentials
//! run as one in-place pass over the uniforms.
//!
//! A pair `(a, b)` gives `r = sqrt(-2 ln(1 - a))` and the normals `r cos(2 pi b)` and
//! `r sin(2 pi b)`, cos first.
//!
//! `1 - a` is exact and in `(0, 1]`. Its exponent bits split it as `m 2^k` with `m` in
//! `[sqrt(1/2), sqrt(2))`, and `ln m = 2 s (1 + z/3 + z^2/5 + ...)` for `s = (m - 1) / (m + 1)`
//! and `z = s^2 <= 0.0295`: a short series that keeps the relative error near the last bit
//! even for `a` close to 0. The angle `2 pi b` needs no range reduction: `b - q/4` for the
//! nearest quarter turn `q` is exact, the series for sine and cosine on `[-pi/4, pi/4]` are
//! short, and a quarter turn is a swap and a sign change on the bits.
//!
//! Every multiply-add is fused, rounded once, as tandem-c does, so that every target gives the
//! same bits. Rust never contracts on its own, so every build does the same arithmetic in the
//! vector body and in the scalar remainder, and a scalar draw equals the fill bit for bit.

/// How a fused multiply-add is computed. The functions below take it as a type parameter,
/// because the same body runs in code compiled for FMA and in code that is not.
pub(crate) trait MulAdd {
    fn fma(a: f64, b: f64, c: f64) -> f64;
    fn fmaf(a: f32, b: f32, c: f32) -> f32;
}

/// The FMA instruction through `mul_add`, for code compiled with it. Elsewhere `mul_add` would
/// be a library call that keeps the loops scalar.
#[cfg(any(
    all(feature = "std", any(target_feature = "fma", target_arch = "aarch64")),
    all(feature = "simd-intrinsics", feature = "std", target_arch = "x86_64")
))]
pub(crate) struct Fused;

#[cfg(any(
    all(feature = "std", any(target_feature = "fma", target_arch = "aarch64")),
    all(feature = "simd-intrinsics", feature = "std", target_arch = "x86_64")
))]
impl MulAdd for Fused {
    #[inline(always)]
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }
    #[inline(always)]
    fn fmaf(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
}

/// A fused multiply-add without the instruction, rounded once all the same, for targets
/// without FMA and for `no_std`, which has no `mul_add`. `fmaf` takes the product exactly in
/// `f64` and rounds the sum to odd, after which rounding to `f32` is correct. `fma` adds
/// Dekker's exact product to `c` by two error-free sums, the last rounded to odd (Boldo and
/// Melquiond, IEEE Trans. Computers 57, 2008). Both equal the instruction bit for bit while
/// nothing underflows or overflows, which the arguments here never do.
// Unused outside the tests where the target has FMA.
#[cfg_attr(
    all(feature = "std", any(target_feature = "fma", target_arch = "aarch64")),
    allow(dead_code)
)]
pub(crate) struct Emulated;

#[cfg_attr(
    all(feature = "std", any(target_feature = "fma", target_arch = "aarch64")),
    allow(dead_code)
)]
impl MulAdd for Emulated {
    #[inline(always)]
    fn fma(a: f64, b: f64, c: f64) -> f64 {
        let split = |x: f64| {
            let t = 134_217_729.0 * x; // 2^27 + 1
            let hi = t - (t - x);
            (hi, x - hi)
        };
        let ((ah, al), (bh, bl)) = (split(a), split(b));
        let p = a * b;
        let pl = ((ah * bh - p) + ah * bl + al * bh) + al * bl; // p + pl = a b
        let (th, tl) = two_sum(c, p);
        let (s, e) = two_sum(tl, pl);
        th + round_to_odd(s, e)
    }
    #[inline(always)]
    fn fmaf(a: f32, b: f32, c: f32) -> f32 {
        let (p, c) = (f64::from(a) * f64::from(b), f64::from(c));
        let (s, e) = two_sum(p, c);
        round_to_odd(s, e) as f32
    }
}

/// `s + e == a + b` exactly.
#[cfg_attr(
    all(feature = "std", any(target_feature = "fma", target_arch = "aarch64")),
    allow(dead_code)
)]
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let t = s - a;
    (s, (a - (s - t)) + (b - t))
}

/// `s` rounded to odd given its error `e`: an inexact sum with an even last bit moves one ulp
/// toward the exact one.
#[cfg_attr(
    all(feature = "std", any(target_feature = "fma", target_arch = "aarch64")),
    allow(dead_code)
)]
#[inline(always)]
fn round_to_odd(s: f64, e: f64) -> f64 {
    let b = s.to_bits();
    let step = if (e > 0.0) == (s > 0.0) { 1 } else { u64::MAX };
    f64::from_bits(if e != 0.0 && b & 1 == 0 {
        b.wrapping_add(step)
    } else {
        b
    })
}

/// The multiply-add of code compiled for the crate's target.
#[cfg(all(feature = "std", any(target_feature = "fma", target_arch = "aarch64")))]
pub(crate) type Native = Fused;
#[cfg(not(all(feature = "std", any(target_feature = "fma", target_arch = "aarch64"))))]
pub(crate) type Native = Emulated;

// The correctly rounded square root, so every target returns the same bits. Without `std` the
// core has no float square root, and `wide` supplies the hardware instruction.
#[cfg(feature = "std")]
#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    wide::f32x4::new([x, 0.0, 0.0, 0.0]).sqrt().to_array()[0]
}

// Coefficients of the series, highest degree first, each step one fused multiply-add.
const LN_SERIES: [f64; 7] = [
    0.08312363319426472,
    0.09070001083303751,
    0.11111433317907482,
    0.14285712049336274,
    0.2000000000566491,
    0.33333333333331017,
    1.0,
];

/// `c[0] x^n + ... + c[n]` by Horner's rule, each step one fused multiply-add.
#[inline(always)]
fn horner<M: MulAdd, const C: usize>(x: f64, c: &[f64; C]) -> f64 {
    let mut acc = c[0];
    for &k in &c[1..] {
        acc = M::fma(x, acc, k);
    }
    acc
}

/// `-2 ln x` for `x` in `(0, 1]`, the reference logarithm `L` of Appendix A.
#[inline(always)]
pub(crate) fn neg2_log<M: MulAdd>(x: f64) -> f64 {
    // Adding the bits of `sqrt(1/2)` to the exponent field makes the mantissa roll over into
    // it exactly when the mantissa is at least `sqrt(1/2)`, which picks `k`.
    let ix = x.to_bits().wrapping_add(0x0009_5f62_0000_0000);
    let nk = (1023 - (ix >> 52) as i64) as f64; // -k
    let mant = f64::from_bits((ix & 0x000f_ffff_ffff_ffff) + 0x3fe6_a09e_0000_0000);
    let s = (mant - 1.0) / (mant + 1.0);
    let p = horner::<M, 7>(s * s, &LN_SERIES);
    // -2 ln x = 2 nk ln 2 - 4 s p, with ln 2 split so that nk * ln2_hi is exact.
    M::fma(
        nk,
        3.816429394731813e-10,
        M::fma(nk, 1.3862943607382476, (s * -4.0) * p),
    )
}

/// `-ln(1 - a)` for `a` in `[0, 1)`. Halving is exact.
#[inline(always)]
pub(crate) fn exponential_f64<M: MulAdd>(a: f64) -> f64 {
    0.5 * neg2_log::<M>(1.0 - a)
}

/// The pair loop unrolled by `U` pairs. The body is then several independent chains, which the
/// vectoriser keeps in flight together so that it loads each constant once for all of them.
// `2 * U` is not a usable const argument of `as_chunks` on stable.
#[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
#[inline(always)]
fn unrolled<T: Copy, const U: usize>(u: &[T], z: &mut [T], pair: fn(T, T) -> [T; 2]) {
    let m = u.len() / 2;
    let (u, z) = (&u[..2 * m], &mut z[..2 * m]);
    let whole = 2 * U * (m / U);
    let (u_whole, u_rest) = u.split_at(whole);
    let (z_whole, z_rest) = z.split_at_mut(whole);
    for (u, z) in u_whole
        .chunks_exact(2 * U)
        .zip(z_whole.chunks_exact_mut(2 * U))
    {
        let pairs: [[T; 2]; U] = core::array::from_fn(|i| pair(u[2 * i], u[2 * i + 1]));
        z.copy_from_slice(pairs.as_flattened());
    }
    for (u, z) in u_rest.chunks_exact(2).zip(z_rest.chunks_exact_mut(2)) {
        z.copy_from_slice(&pair(u[0], u[1]));
    }
}

/// `-2 ln x` in `f32`.
#[inline(always)]
fn neg2_log32<M: MulAdd>(x: f32) -> f32 {
    let ix = x.to_bits().wrapping_add(0x004a_fb0d);
    let nk = (127 - (ix >> 23) as i32) as f32;
    let mant = f32::from_bits((ix & 0x007f_ffff) + 0x3f35_04f3);
    let s = (mant - 1.0) / (mant + 1.0);
    let zz = s * s;
    let p = M::fmaf(
        zz,
        M::fmaf(zz, M::fmaf(zz, 0.14275366, 0.20000061), 0.33333334),
        1.0,
    );
    M::fmaf(nk, 2.857_213_5e-6, M::fmaf(nk, 1.386_291_5, (s * -4.0) * p))
}

/// `sqrt(-2 ln(1 - a))` in `f32`.
#[inline(always)]
pub(crate) fn radius32<M: MulAdd>(a: f32) -> f32 {
    sqrtf(neg2_log32::<M>(1.0 - a))
}

/// `-ln(1 - a)` in `f32`, tandem-c's `neg_log_f32`: within 0.571 ulp for every `a` on the 2^-24
/// grid, so that `1 - exp(-x)` maps each draw back to its own grid point. `u = (2 - 2m) / (m + 1)`
/// is carried as `uh + r / d` with `m + 1 = d + dl` exactly, and `nk ln2_hi + uh` is split by
/// fast two-sum, exact because `nk ln2_hi` is either 0 or larger than `|uh|`.
#[inline(always)]
pub(crate) fn exponential_f32<M: MulAdd>(a: f32) -> f32 {
    let ix = (1.0 - a).to_bits().wrapping_add(0x004a_fb0d);
    let nk = (127 - (ix >> 23) as i32) as f32;
    let mant = f32::from_bits((ix & 0x007f_ffff) + 0x3f35_04f3);
    let num = M::fmaf(mant, -2.0, 2.0);
    let d = mant + 1.0;
    let dl = mant - (d - 1.0);
    let rcp = 1.0 / d;
    // An fma, as in tandem-c, where it keeps a contracting compiler from fusing num * rcp into
    // the two-sum.
    let uh = M::fmaf(num, rcp, 0.0);
    let r = M::fmaf(-uh, dl, M::fmaf(-uh, d, num));
    let v = uh * uh;
    let q = M::fmaf(v, M::fmaf(v, 0.0023109776, 0.012496489), 0.08333336);
    let k_hi = nk * 0.693_145_75;
    let hi = k_hi + uh;
    let e = uh - (hi - k_hi);
    hi + M::fmaf(uh * v, q, M::fmaf(r, rcp, M::fmaf(nk, 1.428_606_8e-6, e)))
}

/// `(cos, sin)` of `2 pi b` in `f32`.
#[inline(always)]
pub(crate) fn rotation32<M: MulAdd>(b: f32) -> (f32, f32) {
    let q = (b * 4.0 + 0.5) as i32;
    let f = M::fmaf(-(q as f32), 0.25, b);
    // 2 pi as a float pair, so that the angle is good to the last bit of the float.
    let th = M::fmaf(f, -1.7484555e-7, f * 6.2831855);
    let w = th * th;
    let hs = M::fmaf(
        w,
        M::fmaf(w, M::fmaf(w, 2.72499e-06, -0.00019840087), 0.008333332),
        -0.16666667,
    );
    let hc = M::fmaf(
        w,
        M::fmaf(w, M::fmaf(w, 2.4463761e-05, -0.0013887589), 0.04166665),
        -0.5,
    );
    let sn = th * M::fmaf(w, hs, 1.0);
    let cs = M::fmaf(w, hc, 1.0);
    let qu = q as u32;
    let sm = 0u32.wrapping_sub(qu & 1);
    let (sb, cb) = (sn.to_bits(), cs.to_bits());
    let xb = ((sb & sm) | (cb & !sm)) ^ ((qu.wrapping_add(1) << 30) & (1 << 31));
    let yb = ((cb & sm) | (sb & !sm)) ^ ((qu << 30) & (1 << 31));
    (f32::from_bits(xb), f32::from_bits(yb))
}

/// One pair of `f32` uniforms to `[cos, sin]` normals.
#[inline(always)]
pub(crate) fn pair_f32<M: MulAdd>(a: f32, b: f32) -> [f32; 2] {
    let r = radius32::<M>(a);
    let (c, s) = rotation32::<M>(b);
    [r * c, r * s]
}

/// Uniforms `[a0, b0, a1, b1, ...]` to normals `[cos0, sin0, cos1, sin1, ...]`. Out of line,
/// with a trip count the compiler does not know, so that it vectorises the loop as one body.
#[inline(never)]
pub(crate) fn block_f32(u: &[f32], z: &mut [f32]) {
    #[cfg(all(feature = "simd-intrinsics", feature = "std", target_arch = "x86_64"))]
    if crate::arch::fma_available() {
        return crate::arch::block_f32_fma(u, z);
    }
    block_f32_body::<Native>(u, z)
}

#[inline(always)]
pub(crate) fn block_f32_body<M: MulAdd>(u: &[f32], z: &mut [f32]) {
    unrolled::<f32, 1>(u, z, pair_f32::<M>)
}

/// `-ln(1 - u)` in place over uniforms. Out of line, so that the vectoriser sees one loop
/// with an unknown trip count, as in [`block_f32`].
#[inline(never)]
pub(crate) fn exponential_block_f64(z: &mut [f64]) {
    #[cfg(all(feature = "simd-intrinsics", feature = "std", target_arch = "x86_64"))]
    if crate::arch::fma_available() {
        return crate::arch::exponential_block_f64_fma(z);
    }
    exponential_block_f64_body::<Native>(z)
}

#[inline(always)]
pub(crate) fn exponential_block_f64_body<M: MulAdd>(z: &mut [f64]) {
    for x in z {
        *x = exponential_f64::<M>(*x);
    }
}

/// The `f32` form of [`exponential_block_f64`].
#[inline(never)]
pub(crate) fn exponential_block_f32(z: &mut [f32]) {
    #[cfg(all(feature = "simd-intrinsics", feature = "std", target_arch = "x86_64"))]
    if crate::arch::fma_available() {
        return crate::arch::exponential_block_f32_fma(z);
    }
    exponential_block_f32_body::<Native>(z)
}

#[inline(always)]
pub(crate) fn exponential_block_f32_body<M: MulAdd>(z: &mut [f32]) {
    for x in z {
        *x = exponential_f32::<M>(*x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tandem;

    // Accuracy of the building blocks against the system functions over random inputs.
    const SAMPLES: usize = 1_000_000;

    // Out of line: inlined into the loops below, the emulated multiply-adds of `no_std` take
    // LLVM minutes to compile.
    #[inline(never)]
    fn log_error(x: f64) -> f64 {
        let want = -2.0 * x.ln();
        ((neg2_log::<Native>(x) - want) / want).abs()
    }

    #[inline(never)]
    fn radius_error(a: f32) -> f64 {
        let want = (-2.0 * (1.0 - f64::from(a)).ln()).sqrt();
        ((f64::from(radius32::<Native>(a)) - want) / want).abs()
    }

    #[test]
    fn logarithm_and_radius_are_within_a_few_ulps() {
        let mut rng = Tandem::new(11);
        let mut worst: f64 = 0.0;
        let mut check = |x: f64| worst = worst.max(log_error(x));
        for _ in 0..SAMPLES {
            check(1.0 - rng.next_f64());
        }
        // Arguments near 1 and near 0 are the hard ends: `ln` cancels at one and the exponent
        // is large at the other.
        for j in 1..=2000 {
            let a = j as f64 * (1.0 / (1u64 << 53) as f64);
            check(1.0 - a);
            check(a);
        }
        assert!(worst < 2e-15, "relative error {worst}");
        let mut worst: f64 = 0.0;
        for _ in 0..SAMPLES {
            worst = worst.max(radius_error(rng.next_f32()));
        }
        assert!(worst < 4e-7, "f32 relative error {worst}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn emulated_fma_rounds_as_the_instruction() {
        // Operands of either sign within 2^±40, and in every other case a c that cancels most
        // of a b, where a sum rounded twice would differ.
        let mut rng = Tandem::new(12);
        let f64_of = |r: &mut Tandem| {
            let x = r.next_u64();
            f64::from_bits((x & 0x800f_ffff_ffff_ffff) | ((1023 - 40 + x % 80) << 52))
        };
        for i in 0..1_000_000 {
            let (a, b) = (f64_of(&mut rng), f64_of(&mut rng));
            let c = if i % 2 == 0 {
                f64_of(&mut rng)
            } else {
                -(a * b) * (1.0 + (rng.next_u32() % 64) as f64 * f64::EPSILON)
            };
            assert_eq!(Emulated::fma(a, b, c).to_bits(), a.mul_add(b, c).to_bits());
            let (x, y) = (a as f32, b as f32);
            let z = if i % 2 == 0 {
                c as f32
            } else {
                -(x * y) * (1.0 + (rng.next_u32() % 16) as f32 * f32::EPSILON)
            };
            assert_eq!(Emulated::fmaf(x, y, z).to_bits(), x.mul_add(y, z).to_bits());
        }
    }
}
