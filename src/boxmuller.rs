//! Box-Muller on whole vectors of pairs, in plain `wide` lane arithmetic.
//!
//! The logarithm is fdlibm's: split `x = 2^k m` with `m` in `[sqrt(1/2), sqrt(2))`, write
//! `s = f / (2 + f)` for `f = m - 1`, and sum a Remez polynomial in `s^2`, which is accurate
//! to under one ulp. The sine and cosine of `2 pi b` need no range reduction: `4b` splits
//! exactly into a quadrant and a fraction of a quarter turn, which `sin(pi/2 - x) = cos(x)`
//! folds into the first octant, where a short Taylor series is exact to rounding.
//!
//! Every operation is lane-wise and Rust does not contract multiply-adds, so a pair gives the
//! same bits whichever lane it sits in. The scalar draws rely on that to equal the fills.

const fn taylor<const N: usize>(first: usize) -> [f64; N] {
    // Coefficients (-1)^k / (first + 2k)!, rounded once from the exact fraction.
    let mut out = [0.0; N];
    let mut k = 0;
    while k < N {
        let mut fact = 1.0;
        let mut i = 2;
        while i <= first + 2 * k {
            fact *= i as f64;
            i += 1;
        }
        out[k] = if k % 2 == 0 { 1.0 / fact } else { -1.0 / fact };
        k += 1;
    }
    out
}

const fn to_f32<const N: usize>(a: [f64; N]) -> [f32; N] {
    let mut out = [0.0; N];
    let mut i = 0;
    while i < N {
        out[i] = a[i] as f32;
        i += 1;
    }
    out
}

// Taylor series on [0, pi/4]: the first omitted term is below 1e-19, and for f32 below 1e-9.
const SIN: [f64; 8] = taylor(1);
const COS: [f64; 9] = taylor(0);
const SIN32: [f32; 5] = to_f32(taylor::<5>(1));
const COS32: [f32; 6] = to_f32(taylor::<6>(0));

// fdlibm e_log.c, digits as published.
#[allow(clippy::excessive_precision)]
const LG: [f64; 7] = [
    6.666666666666735130e-01,
    3.999999999940941908e-01,
    2.857142874366239149e-01,
    2.222219843214978396e-01,
    1.818357216161805012e-01,
    1.531383769920937332e-01,
    1.479819860511658591e-01,
];
#[allow(clippy::excessive_precision)]
const LN2_HI: f64 = 6.931471803691238165e-01;
#[allow(clippy::excessive_precision)]
const LN2_LO: f64 = 1.908214929270587700e-10;

// FreeBSD e_logf.c.
const LG32: [f32; 4] = [
    0xaaaaaa as f32 / (1 << 24) as f32,
    0xccce13 as f32 / (1 << 25) as f32,
    0x91e9ee as f32 / (1 << 25) as f32,
    0xf89e26 as f32 / (1 << 26) as f32,
];
#[allow(clippy::excessive_precision)]
const LN2_HI32: f32 = 6.9313812256e-01;
#[allow(clippy::excessive_precision)]
const LN2_LO32: f32 = 9.0580006145e-06;

/// `a * b + c`, fused where the hardware does it in one instruction: a Horner step is then
/// one vector instruction instead of two. Fusing changes the last bit, so results differ
/// slightly between targets, which the ports' contract allows. Scalar and fill agree within
/// one build because both take this path.
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

// The loops hold more constants than the vector registers, and the compiler rebuilds each one
// from immediates every iteration, which costs several instructions apiece. Passed through
// `black_box` once per call, they are loaded from the stack instead.
struct Consts64 {
    sin: [f64; 8],
    cos: [f64; 9],
    lg: [f64; 7],
    ln2_hi: f64,
    ln2_lo: f64,
    quarter_turn: f64,
}

const CONSTS64: Consts64 = Consts64 {
    sin: SIN,
    cos: COS,
    lg: LG,
    ln2_hi: LN2_HI,
    ln2_lo: LN2_LO,
    quarter_turn: core::f64::consts::FRAC_PI_2,
};

/// The bits of `sqrt(1/2)`: subtracting them turns the exponent field into the exponent of
/// `x / sqrt(1/2)`, so the mantissa lands in `[sqrt(1/2), sqrt(2))` with integer operations
/// only.
const SQRT_HALF_BITS: u64 = 0x3fe6_a09e_667f_3bcd;

/// `ln x` for `x` in `(0, 1]`, normal.
#[inline(always)]
fn ln(x: f64, c: &Consts64) -> f64 {
    let bits = x.to_bits();
    let shifted = bits.wrapping_sub(SQRT_HALF_BITS);
    let k = ((shifted as i64) >> 52) as f64;
    let z = f64::from_bits(bits.wrapping_sub(shifted & (0xfff << 52)));
    let f = z - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let l = &c.lg;
    let t1 = w * fma(w, fma(w, l[5], l[3]), l[1]);
    let t2 = z * fma(w, fma(w, fma(w, l[6], l[4]), l[2]), l[0]);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    k * c.ln2_hi - ((hfsq - (s * (hfsq + r) + k * c.ln2_lo)) - f)
}

/// `(sin, cos)` of `2 pi b` for `b` in `[0, 1)`.
///
/// `4b` rounds to a quarter turn `q` and a remainder `f` in `[-1/2, 1/2]`, both exact, and
/// the Taylor series on `x = f pi/2`, `|x| <= pi/4`, is exact to rounding. The quarter turns
/// then rotate the pair: swap for odd `q`, and a sign flip of the sine for `q mod 4` in
/// `{2, 3}` and of the cosine for `{1, 2}`, all on the bits.
#[inline(always)]
fn sin_cos_2pi(b: f64, c: &Consts64) -> (f64, f64) {
    let t = 4.0 * b;
    let q = (t + 0.5) as u64;
    let x = (t - q as f64) * c.quarter_turn;
    let x2 = x * x;
    let mut s = c.sin[7];
    let mut co = c.cos[8];
    co = fma(co, x2, c.cos[7]);
    for i in (0..7).rev() {
        s = fma(s, x2, c.sin[i]);
        co = fma(co, x2, c.cos[i]);
    }
    let (s0, c0) = ((x * s).to_bits(), co.to_bits());
    let swap = 0u64.wrapping_sub(q & 1) & (s0 ^ c0);
    let (u, v) = (s0 ^ swap, c0 ^ swap);
    (
        f64::from_bits(u ^ ((q & 2) << 62)),
        f64::from_bits(v ^ (((q + 1) & 2) << 62)),
    )
}

/// Pairs per call: enough independent chains to keep the vector units busy through the
/// latency of the polynomials.
pub(crate) const PAIRS: usize = 16;

/// Uniforms `[a0, b0, a1, b1, ...]` to normals `[cos0, sin0, cos1, sin1, ...]`.
#[inline(always)]
pub(crate) fn pairs_f64(d: &[f64; 2 * PAIRS]) -> [f64; 2 * PAIRS] {
    let c = &core::hint::black_box(CONSTS64);
    let mut out = [0.0; 2 * PAIRS];
    for j in 0..PAIRS {
        let r = sqrt(-2.0 * ln(1.0 - d[2 * j], c));
        let (s, co) = sin_cos_2pi(d[2 * j + 1], c);
        out[2 * j] = r * co;
        out[2 * j + 1] = r * s;
    }
    out
}

/// The correctly rounded square root, so every target returns the same bits. Without `std`
/// the core has no float square root, and `wide` supplies the hardware instruction.
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
fn sqrt32(x: f32) -> f32 {
    x.sqrt()
}

#[cfg(not(feature = "std"))]
#[inline(always)]
fn sqrt32(x: f32) -> f32 {
    wide::f32x4::new([x, 0.0, 0.0, 0.0]).sqrt().to_array()[0]
}

/// `a * b + c` in `f32`, as [`fma`].
#[cfg(all(feature = "std", any(target_arch = "aarch64", target_feature = "fma")))]
#[inline(always)]
fn fma32(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg(not(all(feature = "std", any(target_arch = "aarch64", target_feature = "fma"))))]
#[inline(always)]
fn fma32(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

struct Consts32 {
    sin: [f32; 5],
    cos: [f32; 6],
    lg: [f32; 4],
    ln2_hi: f32,
    ln2_lo: f32,
    quarter_turn: f32,
}

const CONSTS32: Consts32 = Consts32 {
    sin: SIN32,
    cos: COS32,
    lg: LG32,
    ln2_hi: LN2_HI32,
    ln2_lo: LN2_LO32,
    quarter_turn: core::f32::consts::FRAC_PI_2,
};

const SQRT_HALF_BITS32: u32 = 0x3f35_04f3;

/// `ln x` for `x` in `(0, 1]`, in `f32`.
#[inline(always)]
fn ln32(x: f32, c: &Consts32) -> f32 {
    let bits = x.to_bits();
    let shifted = bits.wrapping_sub(SQRT_HALF_BITS32);
    let k = ((shifted as i32) >> 23) as f32;
    let z = f32::from_bits(bits.wrapping_sub(shifted & (0x1ff << 23)));
    let f = z - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let l = &c.lg;
    let t1 = w * fma32(w, l[3], l[1]);
    let t2 = z * fma32(w, l[2], l[0]);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    k * c.ln2_hi - ((hfsq - (s * (hfsq + r) + k * c.ln2_lo)) - f)
}

/// `(sin, cos)` of `2 pi b` in `f32`, as [`sin_cos_2pi`].
#[inline(always)]
fn sin_cos_2pi32(b: f32, c: &Consts32) -> (f32, f32) {
    let t = 4.0 * b;
    let q = (t + 0.5) as u32;
    let x = (t - q as f32) * c.quarter_turn;
    let x2 = x * x;
    let mut s = c.sin[4];
    let mut co = c.cos[5];
    co = fma32(co, x2, c.cos[4]);
    for i in (0..4).rev() {
        s = fma32(s, x2, c.sin[i]);
        co = fma32(co, x2, c.cos[i]);
    }
    let (s0, c0) = ((x * s).to_bits(), co.to_bits());
    let swap = 0u32.wrapping_sub(q & 1) & (s0 ^ c0);
    let (u, v) = (s0 ^ swap, c0 ^ swap);
    (
        f32::from_bits(u ^ ((q & 2) << 30)),
        f32::from_bits(v ^ (((q + 1) & 2) << 30)),
    )
}

/// The `f32` form of [`pairs_f64`].
#[inline(always)]
pub(crate) fn pairs_f32(d: &[f32; 2 * PAIRS]) -> [f32; 2 * PAIRS] {
    let c = &core::hint::black_box(CONSTS32);
    let mut out = [0.0; 2 * PAIRS];
    for j in 0..PAIRS {
        let r = sqrt32(-2.0 * ln32(1.0 - d[2 * j], c));
        let (s, co) = sin_cos_2pi32(d[2 * j + 1], c);
        out[2 * j] = r * co;
        out[2 * j + 1] = r * s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tandem;

    // Accuracy of the building blocks against the system functions over random inputs.
    const SAMPLES: usize = 1_000_000;

    #[test]
    fn ln_is_within_two_ulps() {
        let c = &CONSTS64;
        let mut rng = Tandem::new(11);
        let mut worst: f64 = 0.0;
        for _ in 0..SAMPLES {
            let x = 1.0 - rng.next_f64();
            let rel = ((ln(x, c) - x.ln()) / x.ln()).abs();
            worst = worst.max(rel);
        }
        // Uniforms just below 1 make `ln` cancel: the relative error must hold there too.
        for j in 1..=2000 {
            let x = 1.0 - j as f64 * (1.0 / (1u64 << 53) as f64);
            worst = worst.max(((ln(x, c) - x.ln()) / x.ln()).abs());
        }
        assert!(worst < 4.5e-16, "relative error {worst}");
        let c = &CONSTS32;
        let mut worst: f64 = 0.0;
        for _ in 0..SAMPLES {
            let x = 1.0 - rng.next_f32();
            let rel = f64::from((ln32(x, c) - x.ln()) / x.ln()).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 4e-7, "f32 relative error {worst}");
    }

    #[test]
    fn sin_cos_are_within_two_ulps() {
        let c = &CONSTS64;
        let mut rng = Tandem::new(12);
        let mut worst: f64 = 0.0;
        for _ in 0..SAMPLES {
            let b = rng.next_f64();
            let (s, co) = sin_cos_2pi(b, c);
            // The rounded angle of the oracle is off by up to 7e-16.
            let (es, ec) = (core::f64::consts::TAU * b).sin_cos();
            worst = worst.max((s - es).abs()).max((co - ec).abs());
            assert!((s * s + co * co - 1.0).abs() < 6e-16);
        }
        assert!(worst < 1e-15, "absolute error {worst}");
    }
}
