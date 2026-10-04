//! Bounded integers and normals. The specification does not define them. They follow the
//! shared device core (`tandem-cuda`, `core.hpp`) so every port returns the same values.

use crate::Tandem;
use crate::boxmuller::{PAIRS, pairs_f32, pairs_f64};

/// Reserved purposes of the fallback generators of the bounded fills.
const PURPOSE_BELOW32: u64 = 0x0042_4c57_3332;
const PURPOSE_BELOW64: u64 = 0x0042_4c57_3634;

/// Draws per block of a normal fill: a multiple of the 128 `f64` a row holds, so the bulk
/// fill keeps whole rows.
const BLOCK: usize = 4096;

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
        self.normal2_f64()[0]
    }

    /// Two standard normals, `[cos, sin]` halves, from two `f64` draws.
    pub fn normal2_f64(&mut self) -> [f64; 2] {
        let (first, second) = (self.next_f64(), self.next_f64());
        let z = pairs_f64::<2>(&[first, second]);
        [z[0], z[1]]
    }

    /// The cosine half of a Box-Muller pair in `f32` from two `f32` draws, with the same
    /// mapping as [`normal_f64`](Self::normal_f64). Ports agree on it to a few ulps, not bit
    /// for bit: libraries differ in the last bits of the logarithm, sine and cosine.
    pub fn normal_f32(&mut self) -> f32 {
        self.normal2_f32()[0]
    }

    /// Two standard normals, `[cos, sin]` halves, from two `f32` draws.
    pub fn normal2_f32(&mut self) -> [f32; 2] {
        let (first, second) = (self.next_f32(), self.next_f32());
        let z = pairs_f32::<2>(&[first, second]);
        [z[0], z[1]]
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
            normals(draws, chunk, pairs_f64::<{ 2 * PAIRS }>);
        }
    }

    /// The `f32` form of [`fill_normal_f64`](Self::fill_normal_f64), from the `f32` fill.
    pub fn fill_normal_f32(&mut self, out: &mut [f32]) {
        let mut draws = [0.0; BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            let draws = &mut draws[..chunk.len().next_multiple_of(2)];
            self.fill_f32(draws);
            normals(draws, chunk, pairs_f32::<{ 2 * PAIRS }>);
        }
    }
}

/// Convert the uniforms `draws` (twice the pair count) to `out.len()` normals, `N / 2` pairs
/// at a time. The last group is padded with zeros, which are harmless uniforms.
fn normals<T: Copy + Default, const N: usize>(
    draws: &[T],
    out: &mut [T],
    pairs: impl Fn(&[T; N]) -> [T; N],
) {
    let whole = out.len() / N * N;
    let (head, rest) = out.split_at_mut(whole);
    for (d, o) in draws
        .as_chunks::<N>()
        .0
        .iter()
        .zip(head.as_chunks_mut::<N>().0)
    {
        *o = pairs(d);
    }
    if !rest.is_empty() {
        let mut d = [T::default(); N];
        d[..draws.len() - whole].copy_from_slice(&draws[whole..]);
        rest.copy_from_slice(&pairs(&d)[..rest.len()]);
    }
}
