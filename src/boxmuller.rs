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
//! Multiply-adds are fused where the hardware does it in one instruction and plain otherwise.
//! Rust never contracts on its own, so every build does the same arithmetic in the vector
//! body and in the scalar remainder, and a scalar draw equals the fill bit for bit.

// Every multiply-add is an explicit fused one, so that every target gives the same bits as
// tandem-c, which does the same. Without a fused instruction `mul_add` is a correct but slower
// library call. Without `std` there is no `mul_add`, and the plain form differs in the last
// bit.
#[cfg(feature = "std")]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

#[cfg(feature = "std")]
#[inline(always)]
fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

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
fn horner<const C: usize>(x: f64, c: &[f64; C]) -> f64 {
    let mut acc = c[0];
    for &k in &c[1..] {
        acc = fma(x, acc, k);
    }
    acc
}

/// `-2 ln x` for `x` in `(0, 1]`, the reference logarithm `L` of Appendix A.
#[inline(always)]
pub(crate) fn neg2_log(x: f64) -> f64 {
    // Adding the bits of `sqrt(1/2)` to the exponent field makes the mantissa roll over into
    // it exactly when the mantissa is at least `sqrt(1/2)`, which picks `k`.
    let ix = x.to_bits().wrapping_add(0x0009_5f62_0000_0000);
    let nk = (1023 - (ix >> 52) as i64) as f64; // -k
    let mant = f64::from_bits((ix & 0x000f_ffff_ffff_ffff) + 0x3fe6_a09e_0000_0000);
    let s = (mant - 1.0) / (mant + 1.0);
    let p = horner(s * s, &LN_SERIES);
    // -2 ln x = 2 nk ln 2 - 4 s p, with ln 2 split so that nk * ln2_hi is exact.
    fma(
        nk,
        3.816429394731813e-10,
        fma(nk, 1.3862943607382476, (s * -4.0) * p),
    )
}

/// `-ln(1 - a)` for `a` in `[0, 1)`. Halving is exact.
#[inline(always)]
pub(crate) fn exponential_f64(a: f64) -> f64 {
    0.5 * neg2_log(1.0 - a)
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
fn neg2_log32(x: f32) -> f32 {
    let ix = x.to_bits().wrapping_add(0x004a_fb0d);
    let nk = (127 - (ix >> 23) as i32) as f32;
    let mant = f32::from_bits((ix & 0x007f_ffff) + 0x3f35_04f3);
    let s = (mant - 1.0) / (mant + 1.0);
    let zz = s * s;
    let p = fmaf(
        zz,
        fmaf(zz, fmaf(zz, 0.14275366, 0.20000061), 0.33333334),
        1.0,
    );
    fmaf(nk, 2.857_213_5e-6, fmaf(nk, 1.386_291_5, (s * -4.0) * p))
}

/// `sqrt(-2 ln(1 - a))` in `f32`.
#[inline(always)]
pub(crate) fn radius32(a: f32) -> f32 {
    sqrtf(neg2_log32(1.0 - a))
}

/// `-ln(1 - a)` in `f32`, tandem-c's `neg_log_f32`: within 0.571 ulp for every `a` on the 2^-24
/// grid, so that `1 - exp(-x)` maps each draw back to its own grid point. `u = (2 - 2m) / (m + 1)`
/// is carried as `uh + r / d` with `m + 1 = d + dl` exactly, and `nk ln2_hi + uh` is split by
/// fast two-sum, exact because `nk ln2_hi` is either 0 or larger than `|uh|`.
#[inline(always)]
pub(crate) fn exponential_f32(a: f32) -> f32 {
    let ix = (1.0 - a).to_bits().wrapping_add(0x004a_fb0d);
    let nk = (127 - (ix >> 23) as i32) as f32;
    let mant = f32::from_bits((ix & 0x007f_ffff) + 0x3f35_04f3);
    let num = fmaf(mant, -2.0, 2.0);
    let d = mant + 1.0;
    let dl = mant - (d - 1.0);
    let rcp = 1.0 / d;
    // An fma, as in tandem-c, where it keeps a contracting compiler from fusing num * rcp into
    // the two-sum.
    let uh = fmaf(num, rcp, 0.0);
    let r = fmaf(-uh, dl, fmaf(-uh, d, num));
    let v = uh * uh;
    let q = fmaf(v, fmaf(v, 0.0023109776, 0.012496489), 0.08333336);
    let k_hi = nk * 0.693_145_75;
    let hi = k_hi + uh;
    let e = uh - (hi - k_hi);
    hi + fmaf(uh * v, q, fmaf(r, rcp, fmaf(nk, 1.428_606_8e-6, e)))
}

/// `(cos, sin)` of `2 pi b` in `f32`.
#[inline(always)]
pub(crate) fn rotation32(b: f32) -> (f32, f32) {
    let q = (b * 4.0 + 0.5) as i32;
    let f = fmaf(-(q as f32), 0.25, b);
    // 2 pi as a float pair, so that the angle is good to the last bit of the float.
    let th = fmaf(f, -1.7484555e-7, f * 6.2831855);
    let w = th * th;
    let hs = fmaf(
        w,
        fmaf(w, fmaf(w, 2.72499e-06, -0.00019840087), 0.008333332),
        -0.16666667,
    );
    let hc = fmaf(
        w,
        fmaf(w, fmaf(w, 2.4463761e-05, -0.0013887589), 0.04166665),
        -0.5,
    );
    let sn = th * fmaf(w, hs, 1.0);
    let cs = fmaf(w, hc, 1.0);
    let qu = q as u32;
    let sm = 0u32.wrapping_sub(qu & 1);
    let (sb, cb) = (sn.to_bits(), cs.to_bits());
    let xb = ((sb & sm) | (cb & !sm)) ^ ((qu.wrapping_add(1) << 30) & (1 << 31));
    let yb = ((cb & sm) | (sb & !sm)) ^ ((qu << 30) & (1 << 31));
    (f32::from_bits(xb), f32::from_bits(yb))
}

/// One pair of `f32` uniforms to `[cos, sin]` normals.
#[inline(always)]
pub(crate) fn pair_f32(a: f32, b: f32) -> [f32; 2] {
    let r = radius32(a);
    let (c, s) = rotation32(b);
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
    block_f32_body(u, z)
}

#[inline(always)]
pub(crate) fn block_f32_body(u: &[f32], z: &mut [f32]) {
    unrolled::<f32, 1>(u, z, pair_f32)
}

/// `-ln(1 - u)` in place over uniforms. Out of line, so that the vectoriser sees one loop
/// with an unknown trip count, as in [`block_f32`].
#[inline(never)]
pub(crate) fn exponential_block_f64(z: &mut [f64]) {
    #[cfg(all(feature = "simd-intrinsics", feature = "std", target_arch = "x86_64"))]
    if crate::arch::fma_available() {
        return crate::arch::exponential_block_f64_fma(z);
    }
    exponential_block_f64_body(z)
}

#[inline(always)]
pub(crate) fn exponential_block_f64_body(z: &mut [f64]) {
    for x in z {
        *x = exponential_f64(*x);
    }
}

/// The `f32` form of [`exponential_block_f64`].
#[inline(never)]
pub(crate) fn exponential_block_f32(z: &mut [f32]) {
    #[cfg(all(feature = "simd-intrinsics", feature = "std", target_arch = "x86_64"))]
    if crate::arch::fma_available() {
        return crate::arch::exponential_block_f32_fma(z);
    }
    exponential_block_f32_body(z)
}

#[inline(always)]
pub(crate) fn exponential_block_f32_body(z: &mut [f32]) {
    for x in z {
        *x = exponential_f32(*x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tandem;

    // Accuracy of the building blocks against the system functions over random inputs.
    const SAMPLES: usize = 1_000_000;

    #[test]
    fn logarithm_and_radius_are_within_a_few_ulps() {
        let mut rng = Tandem::new(11);
        let mut worst: f64 = 0.0;
        let mut check = |x: f64| {
            let want = -2.0 * x.ln();
            worst = worst.max(((neg2_log(x) - want) / want).abs());
        };
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
            let a = rng.next_f32();
            let want = (-2.0 * (1.0 - f64::from(a)).ln()).sqrt();
            worst = worst.max(((f64::from(radius32(a)) - want) / want).abs());
        }
        assert!(worst < 4e-7, "f32 relative error {worst}");
    }
}
