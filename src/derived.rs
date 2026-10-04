//! Bounded integers and normals. The specification does not define them. They follow the
//! shared device core (`tandem-cuda`, `core.hpp`) so every port returns the same values.

use crate::Tandem;

/// Reserved purposes of the fallback generators of the bounded fills.
const PURPOSE_BELOW32: u64 = 0x0042_4c57_3332;
const PURPOSE_BELOW64: u64 = 0x0042_4c57_3634;

/// Draws per block of a normal fill: a multiple of the 128 `f64` a row holds, so the bulk
/// fill keeps whole rows.
const BLOCK: usize = 256;

impl Tandem {
    /// A uniform integer in `0..n` by Lemire's multiply and reject on `u32` draws.
    ///
    /// For `n == 0` the result is 0 after one draw, as the device core does.
    pub fn below_u32(&mut self, n: u32) -> u32 {
        let mut m = u64::from(self.next_u32()) * u64::from(n);
        if (m as u32) < n {
            let t = n.wrapping_neg() % n;
            while (m as u32) < t {
                m = u64::from(self.next_u32()) * u64::from(n);
            }
        }
        (m >> 32) as u32
    }

    /// A uniform integer in `0..n` by Lemire's multiply and reject on `u64` draws.
    ///
    /// For `n == 0` the result is 0 after one draw, as the device core does.
    pub fn below_u64(&mut self, n: u64) -> u64 {
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        if (m as u64) < n {
            let t = n.wrapping_neg() % n;
            while (m as u64) < t {
                m = u128::from(self.next_u64()) * u128::from(n);
            }
        }
        (m >> 64) as u64
    }

    /// Fill with uniform integers in `0..n`. Element `i` takes draw `i` of the `u32` fill and
    /// the fill consumes exactly `out.len()` draws, whatever is rejected, so rows fill
    /// independently. A rejected draw retries with Lemire's rule on the `u32` draws of the
    /// generator `from_key(key, 0, K).sub(PURPOSE_BELOW32).split(i)`. A fill without
    /// rejections equals the scalar [`below_u32`](Self::below_u32) calls.
    pub fn fill_below_u32(&mut self, out: &mut [u32], n: u32) {
        self.fill_u32(out);
        self.bound_u32(out, 0, n);
    }

    /// Fill with uniform integers in `0..n` from the `u64` fill, as
    /// [`fill_below_u32`](Self::fill_below_u32) does from the `u32` fill, with
    /// `PURPOSE_BELOW64`.
    pub fn fill_below_u64(&mut self, out: &mut [u64], n: u64) {
        self.fill_u64(out);
        self.bound_u64(out, 0, n);
    }

    /// Map the raw draws of elements `first..` to bounded values in place.
    pub(crate) fn bound_u32(&self, raw: &mut [u32], first: u64, n: u32) {
        for (e, x) in (first..).zip(raw) {
            let m = u64::from(*x) * u64::from(n);
            *x = if (m as u32) < n && (m as u32) < n.wrapping_neg() % n {
                self.retry_u32(n, e)
            } else {
                (m >> 32) as u32
            };
        }
    }

    pub(crate) fn bound_u64(&self, raw: &mut [u64], first: u64, n: u64) {
        for (e, x) in (first..).zip(raw) {
            let m = u128::from(*x) * u128::from(n);
            *x = if (m as u64) < n && (m as u64) < n.wrapping_neg() % n {
                self.retry_u64(n, e)
            } else {
                (m >> 64) as u64
            };
        }
    }

    // Rare, and seeding a generator is costly, so keep it off the common path.
    #[cold]
    #[inline(never)]
    fn retry_u32(&self, n: u32, e: u64) -> u32 {
        let mut r = Tandem::from_key(self.key, 0, self.k)
            .sub(PURPOSE_BELOW32)
            .split(e);
        let t = n.wrapping_neg() % n;
        loop {
            let m = u64::from(r.next_u32()) * u64::from(n);
            if (m as u32) >= t {
                return (m >> 32) as u32;
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn retry_u64(&self, n: u64, e: u64) -> u64 {
        let mut r = Tandem::from_key(self.key, 0, self.k)
            .sub(PURPOSE_BELOW64)
            .split(e);
        let t = n.wrapping_neg() % n;
        loop {
            let m = u128::from(r.next_u64()) * u128::from(n);
            if (m as u64) >= t {
                return (m >> 64) as u64;
            }
        }
    }

    /// The cosine half of a Box-Muller pair from two `f64` draws: `u = 1 - first` lies in
    /// `(0, 1]`, so the logarithm is finite. It equals element 0 of
    /// [`normal2_f64`](Self::normal2_f64).
    pub fn normal_f64(&mut self) -> f64 {
        let first = self.next_f64();
        let second = self.next_f64();
        box_muller2(first, second)[0]
    }

    /// Two standard normals, `[cos, sin]` halves, from two `f64` draws.
    pub fn normal2_f64(&mut self) -> [f64; 2] {
        let first = self.next_f64();
        let second = self.next_f64();
        box_muller2(first, second)
    }

    /// The cosine half of a Box-Muller pair in `f32` from two `f32` draws, with the same
    /// mapping as [`normal_f64`](Self::normal_f64). Ports agree on it to a few ulps, not bit
    /// for bit: `logf`, `cosf` and `sinf` differ between libraries.
    pub fn normal_f32(&mut self) -> f32 {
        let first = self.next_f32();
        let second = self.next_f32();
        box_muller2_f32(first, second)[0]
    }

    /// Two standard normals, `[cos, sin]` halves, from two `f32` draws.
    pub fn normal2_f32(&mut self) -> [f32; 2] {
        let first = self.next_f32();
        let second = self.next_f32();
        box_muller2_f32(first, second)
    }

    /// Fill with standard normals. Pair `j` is elements `2j` and `2j + 1`, cos half first,
    /// from draws `2j` and `2j + 1` of the `f64` fill, so the fill is the flattened
    /// [`normal2_f64`](Self::normal2_f64) sequence. An odd length uses the cos half of its
    /// last pair and still consumes both draws.
    pub fn fill_normal_f64(&mut self, out: &mut [f64]) {
        let mut draws = [0.0; BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            let draws = &mut draws[..chunk.len().next_multiple_of(2)];
            self.fill_f64(draws);
            for (x, pair) in chunk.chunks_mut(2).zip(draws.as_chunks::<2>().0) {
                let z = box_muller2(pair[0], pair[1]);
                x.copy_from_slice(&z[..x.len()]);
            }
        }
    }

    /// The `f32` form of [`fill_normal_f64`](Self::fill_normal_f64), from the `f32` fill.
    pub fn fill_normal_f32(&mut self, out: &mut [f32]) {
        let mut draws = [0.0; BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            let draws = &mut draws[..chunk.len().next_multiple_of(2)];
            self.fill_f32(draws);
            for (x, pair) in chunk.chunks_mut(2).zip(draws.as_chunks::<2>().0) {
                let z = box_muller2_f32(pair[0], pair[1]);
                x.copy_from_slice(&z[..x.len()]);
            }
        }
    }
}

// The logarithm and square root come from the system libm with `std`, which is several times
// faster than the portable `libm` crate, and from that crate without. Both are accurate to an
// ulp or so, and the ports agree on normals to that, not bit for bit.
#[cfg(feature = "std")]
mod math {
    pub fn sqrt(x: f64) -> f64 {
        x.sqrt()
    }
    pub fn ln(x: f64) -> f64 {
        x.ln()
    }
    pub fn sqrtf(x: f32) -> f32 {
        x.sqrt()
    }
    pub fn lnf(x: f32) -> f32 {
        x.ln()
    }
}

#[cfg(not(feature = "std"))]
mod math {
    pub use libm::{sqrt, sqrtf};
    pub fn ln(x: f64) -> f64 {
        libm::log(x)
    }
    pub fn lnf(x: f32) -> f32 {
        libm::logf(x)
    }
}

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

// Taylor series on [0, pi/4]: the first omitted term is below 1e-19.
const SIN: [f64; 9] = taylor(1);
const COS: [f64; 9] = taylor(0);

/// `(sin, cos)` of `2 pi b` for `b` in `[0, 1)`. The angle needs no range reduction: `4b`
/// splits exactly into a quadrant and a fraction of a quarter turn, which the identity
/// `sin(pi/2 - x) = cos(x)` folds into the first octant. A float `2 pi b` would be off by up
/// to `2 pi b 2^-53`, so this is also the more accurate way.
#[inline(always)]
fn sin_cos_2pi(b: f64) -> (f64, f64) {
    let t = 4.0 * b;
    let q = t as u32;
    let f = t - f64::from(q);
    let flip = f > 0.5;
    let x = if flip { 1.0 - f } else { f } * core::f64::consts::FRAC_PI_2;
    let x2 = x * x;
    let horner = |c: &[f64; 9]| c.iter().rev().fold(0.0, |acc, &k| acc * x2 + k);
    let (s, c) = (x * horner(&SIN), horner(&COS));
    let (s, c) = if flip { (c, s) } else { (s, c) };
    match q & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

fn box_muller2(first: f64, second: f64) -> [f64; 2] {
    let r = math::sqrt(-2.0 * math::ln(1.0 - first));
    let (s, c) = sin_cos_2pi(second);
    [r * c, r * s]
}

fn box_muller2_f32(first: f32, second: f32) -> [f32; 2] {
    let r = math::sqrtf(-2.0 * math::lnf(1.0 - first));
    // The angle goes through f64, as the device core does on a host.
    let (s, c) = sin_cos_2pi(f64::from(second));
    [r * c as f32, r * s as f32]
}
