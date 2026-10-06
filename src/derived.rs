//! Bounded integers, `f32` normals and exponentials. The specification defines them in its
//! non-normative Appendix A, so every port returns the same values. The `f64` normals are in
//! `ziggurat`.

use crate::Tandem;
use crate::boxmuller::{
    block_f32, exponential_block_f32, exponential_block_f64, exponential_f32, exponential_f64,
    pair_f32,
};

/// Reserved purposes of the fallback generators of the bounded fills.
const PURPOSE_BELOW32: u64 = 0x0042_4c57_3332;
const PURPOSE_BELOW64: u64 = 0x0042_4c57_3634;

/// Draws per block of a normal fill: a multiple of the 256 `f32` a row holds, so the bulk
/// fill keeps whole rows.
const BLOCK: usize = 4096;

/// Elements per pass of an exponential fill: the uniforms and their map stay in L1.
const EXP_BLOCK: usize = 1024;

impl Tandem {
    /// A uniform integer in `0..n` by Lemire's multiply and reject on `u32` draws.
    ///
    /// For `n == 0` the result is 0 after one draw, as the device core does.
    #[inline]
    pub fn below_u32(&mut self, n: u32) -> u32 {
        below_u32(n, || self.next_u32())
    }

    /// A uniform integer in `0..n` by Lemire's multiply and reject on `u64` draws.
    ///
    /// For `n == 0` the result is 0 after one draw, as the device core does.
    #[inline]
    pub fn below_u64(&mut self, n: u64) -> u64 {
        below_u64(n, || self.next_u64())
    }

    /// Fill with uniform integers in `0..n`. Element `i` takes draw `i` of the `u32` fill,
    /// which has global draw index `g`, the aligned start position over 32 plus `i`. The fill
    /// consumes exactly `out.len()` draws, whatever is rejected, so rows fill independently.
    /// A rejected draw retries with Lemire's rule on the `u32` draws of the generator
    /// `from_key(key, 0, K).sub(PURPOSE_BELOW32).split(g)`, so a fill cut anywhere equals the
    /// whole. A fill without rejections equals the scalar [`below_u32`](Self::below_u32) calls.
    pub fn fill_below_u32(&mut self, out: &mut [u32], n: u32) {
        // Appendix A: an empty bounded fill leaves the position where it is.
        if out.is_empty() {
            return;
        }
        let first = crate::align(self.pos, 32) / 32;
        self.fill_u32(out);
        self.bound_u32(out, first, n);
    }

    /// Fill with uniform integers in `0..n` from the `u64` fill, as
    /// [`fill_below_u32`](Self::fill_below_u32) does from the `u32` fill, with
    /// `PURPOSE_BELOW64`.
    pub fn fill_below_u64(&mut self, out: &mut [u64], n: u64) {
        if out.is_empty() {
            return;
        }
        let first = crate::align(self.pos, 64) / 64;
        self.fill_u64(out);
        self.bound_u64(out, first, n);
    }

    /// Map the raw draws, whose global draw indices start at `first`, to bounded values in place.
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

    /// The cosine half of a Box-Muller pair in `f32` from two `f32` draws: `u = 1 - first`
    /// lies in `(0, 1]`, so the logarithm is finite. With `std` it matches tandem-c bit for
    /// bit. Ports with other `log`, `cos` and `sin` agree to a few ulps.
    #[inline]
    pub fn normal_f32(&mut self) -> f32 {
        self.normal2_f32()[0]
    }

    /// Two standard normals, `[cos, sin]` halves, from two `f32` draws.
    #[inline]
    pub fn normal2_f32(&mut self) -> [f32; 2] {
        let (first, second) = (self.next_f32(), self.next_f32());
        pair_f32(first, second)
    }

    /// Fill with standard normals. Pair `j` is elements `2j` and `2j + 1`, cos half first,
    /// from draws `2j` and `2j + 1` of the `f32` fill, so the fill is the flattened
    /// [`normal2_f32`](Self::normal2_f32) sequence. An odd length uses the cos half of its
    /// last pair and still consumes both draws.
    pub fn fill_normal_f32(&mut self, out: &mut [f32]) {
        let mut draws = [0.0; BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            let draws = &mut draws[..chunk.len().next_multiple_of(2)];
            self.fill_f32(draws);
            normals(draws, chunk, block_f32, pair_f32);
        }
    }

    /// An exponential `-ln(1 - u)` of rate 1 from one `f64` draw. It equals element 0 of
    /// [`fill_exponential_f64`](Self::fill_exponential_f64).
    #[inline]
    pub fn exponential_f64(&mut self) -> f64 {
        exponential_f64(self.next_f64())
    }

    /// The `f32` form of [`exponential_f64`](Self::exponential_f64), from one `f32` draw and
    /// computed in `f32`. It matches tandem-c bit for bit.
    #[inline]
    pub fn exponential_f32(&mut self) -> f32 {
        exponential_f32(self.next_f32())
    }

    /// Fill with exponentials of rate 1. Element `i` is `-ln(1 - u)` for draw `i` of the
    /// `f64` fill, so a fill cut anywhere equals the whole and equals the scalar draws.
    pub fn fill_exponential_f64(&mut self, out: &mut [f64]) {
        for chunk in out.chunks_mut(EXP_BLOCK) {
            self.fill_f64(chunk);
            exponential_block_f64(chunk);
        }
    }

    /// The `f32` form of [`fill_exponential_f64`](Self::fill_exponential_f64), from the `f32`
    /// fill.
    pub fn fill_exponential_f32(&mut self, out: &mut [f32]) {
        for chunk in out.chunks_mut(EXP_BLOCK) {
            self.fill_f32(chunk);
            exponential_block_f32(chunk);
        }
    }
}

/// Lemire's multiply and reject on the draws of `next`.
#[inline]
pub(crate) fn below_u32(n: u32, mut next: impl FnMut() -> u32) -> u32 {
    let mut m = u64::from(next()) * u64::from(n);
    if (m as u32) < n {
        let t = n.wrapping_neg() % n;
        while (m as u32) < t {
            m = u64::from(next()) * u64::from(n);
        }
    }
    (m >> 32) as u32
}

#[inline]
pub(crate) fn below_u64(n: u64, mut next: impl FnMut() -> u64) -> u64 {
    let mut m = u128::from(next()) * u128::from(n);
    if (m as u64) < n {
        let t = n.wrapping_neg() % n;
        while (m as u64) < t {
            m = u128::from(next()) * u128::from(n);
        }
    }
    (m >> 64) as u64
}

/// Convert the uniforms `draws` (twice the pair count) to `out.len()` normals. An odd last
/// element is the cos half of a whole pair.
fn normals<T: Copy>(
    draws: &[T],
    out: &mut [T],
    block: fn(&[T], &mut [T]),
    pair: fn(T, T) -> [T; 2],
) {
    let even = out.len() / 2 * 2;
    block(&draws[..even], &mut out[..even]);
    if even < out.len() {
        out[even] = pair(draws[even], draws[even + 1])[0];
    }
}
