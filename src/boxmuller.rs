//! Box-Muller on whole blocks of pairs, in plain Rust that LLVM vectorises. The shape and the
//! coefficients are tandem-c's, so the ports compute the same arithmetic.
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

#[cfg(all(feature = "std", any(target_arch = "aarch64", target_feature = "fma")))]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[cfg(not(all(feature = "std", any(target_arch = "aarch64", target_feature = "fma"))))]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

#[cfg(all(feature = "std", any(target_arch = "aarch64", target_feature = "fma")))]
#[inline(always)]
fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg(not(all(feature = "std", any(target_arch = "aarch64", target_feature = "fma"))))]
#[inline(always)]
fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

// The correctly rounded square root, so every target returns the same bits. Without `std` the
// core has no float square root, and `wide` supplies the hardware instruction.
#[cfg(feature = "std")]
#[inline(always)]
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn sqrt(x: f64) -> f64 {
    wide::f64x2::new([x, 0.0]).sqrt().to_array()[0]
}

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
const SIN_SERIES: [f64; 6] = [
    1.5914650986900946e-10,
    -2.5051097984389413e-08,
    2.755731600073921e-06,
    -0.00019841269836630226,
    0.008333333333330813,
    -0.16666666666666669,
];
const COS_SERIES: [f64; 6] = [
    2.0665708703855164e-09,
    -2.7555858522576447e-07,
    2.480158263811954e-05,
    -0.0013888888882156126,
    0.04166666666663108,
    -0.4999999999999997,
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

/// `sqrt(-2 ln(1 - a))` for `a` in `[0, 1)`.
#[inline(always)]
fn radius(a: f64) -> f64 {
    // Adding the bits of `sqrt(1/2)` to the exponent field makes the mantissa roll over into
    // it exactly when the mantissa is at least `sqrt(1/2)`, which picks `k`.
    let ix = (1.0 - a).to_bits().wrapping_add(0x0009_5f62_0000_0000);
    let nk = (1023 - (ix >> 52) as i64) as f64; // -k
    let mant = f64::from_bits((ix & 0x000f_ffff_ffff_ffff) + 0x3fe6_a09e_0000_0000);
    let s = (mant - 1.0) / (mant + 1.0);
    let p = horner(s * s, &LN_SERIES);
    // -2 ln(1 - a) = 2 nk ln 2 - 4 s p, with ln 2 split so that nk * ln2_hi is exact.
    sqrt(fma(nk, 1.3862943607382476, (s * -4.0) * p) + nk * 3.816429394731813e-10)
}

/// `(cos, sin)` of `2 pi b` for `b` in `[0, 1)`.
#[inline(always)]
fn rotation(b: f64) -> (f64, f64) {
    let q = (b * 4.0 + 0.5) as i64;
    let th = fma(-(q as f64), 0.25, b) * core::f64::consts::TAU;
    let w = th * th;
    let sn = th * fma(w, horner(w, &SIN_SERIES), 1.0);
    let cs = fma(w, horner(w, &COS_SERIES), 1.0);
    // Odd q swaps the two, bit 1 of q negates the sine, and bit 1 of q + 1 negates the cosine.
    let qu = q as u64;
    let sm = 0u64.wrapping_sub(qu & 1);
    let (sb, cb) = (sn.to_bits(), cs.to_bits());
    let xb = ((sb & sm) | (cb & !sm)) ^ ((qu.wrapping_add(1) << 62) & (1 << 63));
    let yb = ((cb & sm) | (sb & !sm)) ^ ((qu << 62) & (1 << 63));
    (f64::from_bits(xb), f64::from_bits(yb))
}

/// One pair of uniforms to `[cos, sin]` normals.
#[inline(always)]
pub(crate) fn pair_f64(a: f64, b: f64) -> [f64; 2] {
    let r = radius(a);
    let (c, s) = rotation(b);
    [r * c, r * s]
}

/// Uniforms `[a0, b0, a1, b1, ...]` to normals `[cos0, sin0, cos1, sin1, ...]`. Out of line,
/// with a trip count the compiler does not know, so that it vectorises and interleaves the
/// loop as one body.
#[inline(never)]
pub(crate) fn block_f64(u: &[f64], z: &mut [f64]) {
    unrolled::<f64, 4>(u, z, pair_f64)
}

/// The pair loop unrolled by `U` pairs. The body is then several independent chains, which the
/// vectoriser keeps in flight together so that it loads each constant once for all of them.
// `2 * U` is not a usable const argument of `as_chunks` on stable.
#[allow(clippy::chunks_exact_to_as_chunks)]
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

/// `sqrt(-2 ln(1 - a))` in `f32`.
#[inline(always)]
pub(crate) fn radius32(a: f32) -> f32 {
    let ix = (1.0 - a).to_bits().wrapping_add(0x004a_fb0d);
    let nk = (127 - (ix >> 23) as i32) as f32;
    let mant = f32::from_bits((ix & 0x007f_ffff) + 0x3f35_04f3);
    let s = (mant - 1.0) / (mant + 1.0);
    let zz = s * s;
    let p = fmaf(
        zz,
        fmaf(zz, fmaf(zz, 0.14275366, 0.20000061), 0.33333334),
        1.0,
    );
    sqrtf(fmaf(nk, 1.386_291_5, (s * -4.0) * p) + nk * 2.857_213_5e-6)
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

/// The `f32` form of [`block_f64`].
#[inline(never)]
pub(crate) fn block_f32(u: &[f32], z: &mut [f32]) {
    unrolled::<f32, 1>(u, z, pair_f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tandem;

    // Accuracy of the building blocks against the system functions over random inputs.
    const SAMPLES: usize = 1_000_000;

    #[test]
    fn radius_is_within_a_few_ulps() {
        let mut rng = Tandem::new(11);
        let mut worst: f64 = 0.0;
        for _ in 0..SAMPLES {
            let a = rng.next_f64();
            let want = (-2.0 * (1.0 - a).ln()).sqrt();
            worst = worst.max(((radius(a) - want) / want).abs());
        }
        // Uniforms near 0 and just below 1 are the hard ends: `ln` cancels at one and the
        // exponent is large at the other.
        for j in 1..=2000 {
            let a = j as f64 * (1.0 / (1u64 << 53) as f64);
            let want = (-2.0 * (1.0 - a).ln()).sqrt();
            worst = worst.max(((radius(a) - want) / want).abs());
            let a = 1.0 - a;
            let want = (-2.0 * (1.0 - a).ln()).sqrt();
            worst = worst.max(((radius(a) - want) / want).abs());
        }
        assert!(worst < 1e-15, "relative error {worst}");
        let mut worst: f64 = 0.0;
        for _ in 0..SAMPLES {
            let a = rng.next_f32();
            let want = (-2.0 * (1.0 - f64::from(a)).ln()).sqrt();
            worst = worst.max(((f64::from(radius32(a)) - want) / want).abs());
        }
        assert!(worst < 4e-7, "f32 relative error {worst}");
    }

    #[test]
    fn rotation_is_within_a_few_ulps() {
        let mut rng = Tandem::new(12);
        let mut worst: f64 = 0.0;
        for _ in 0..SAMPLES {
            let b = rng.next_f64();
            let (c, s) = rotation(b);
            // The rounded angle of the oracle is off by up to 7e-16.
            let (es, ec) = (core::f64::consts::TAU * b).sin_cos();
            worst = worst.max((s - es).abs()).max((c - ec).abs());
            assert!((s * s + c * c - 1.0).abs() < 1.5e-15);
        }
        assert!(worst < 1e-15, "absolute error {worst}");
    }
}
