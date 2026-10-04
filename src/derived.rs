//! Bounded integers and normals. The specification does not define them. They follow the
//! shared device core (`tandem-cuda`, `core.hpp`) so every port returns the same values.

use crate::Tandem;

use core::f64::consts::TAU;

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

    /// Fill with the values of `out.len()` calls to [`below_u32`](Self::below_u32). Rejection
    /// makes the draw count data dependent, so this is a loop of scalar draws.
    pub fn fill_below_u32(&mut self, out: &mut [u32], n: u32) {
        for x in out {
            *x = self.below_u32(n);
        }
    }

    /// Fill with the values of `out.len()` calls to [`below_u64`](Self::below_u64).
    pub fn fill_below_u64(&mut self, out: &mut [u64], n: u64) {
        for x in out {
            *x = self.below_u64(n);
        }
    }

    /// A standard normal by Box-Muller from two `f64` draws: `u = 1 - first` lies in
    /// `(0, 1]`, so the logarithm is finite.
    pub fn normal_f64(&mut self) -> f64 {
        let first = self.next_f64();
        let second = self.next_f64();
        box_muller(first, second)
    }

    /// A standard normal by Box-Muller in `f32` from two `f32` draws, with the same mapping
    /// as [`normal_f64`](Self::normal_f64). Ports agree on it to a few ulps, not bit for
    /// bit: `logf` and `cosf` differ between libraries.
    pub fn normal_f32(&mut self) -> f32 {
        let first = self.next_f32();
        let second = self.next_f32();
        box_muller_f32(first, second)
    }

    /// Fill with standard normals: element `i` uses the `f64` draws `2i` and `2i + 1`, as
    /// `out.len()` calls to [`normal_f64`](Self::normal_f64) would.
    pub fn fill_normal_f64(&mut self, out: &mut [f64]) {
        let mut draws = [0.0; BLOCK];
        for chunk in out.chunks_mut(BLOCK / 2) {
            let draws = &mut draws[..2 * chunk.len()];
            self.fill_f64(draws);
            for (x, pair) in chunk.iter_mut().zip(draws.as_chunks::<2>().0) {
                *x = box_muller(pair[0], pair[1]);
            }
        }
    }

    /// Fill with standard normals: element `i` uses the `f32` draws `2i` and `2i + 1`, as
    /// `out.len()` calls to [`normal_f32`](Self::normal_f32) would.
    pub fn fill_normal_f32(&mut self, out: &mut [f32]) {
        let mut draws = [0.0; BLOCK];
        for chunk in out.chunks_mut(BLOCK / 2) {
            let draws = &mut draws[..2 * chunk.len()];
            self.fill_f32(draws);
            for (x, pair) in chunk.iter_mut().zip(draws.as_chunks::<2>().0) {
                *x = box_muller_f32(pair[0], pair[1]);
            }
        }
    }
}

#[inline]
fn box_muller(first: f64, second: f64) -> f64 {
    libm::sqrt(-2.0 * libm::log(1.0 - first)) * libm::cos(TAU * second)
}

#[inline]
fn box_muller_f32(first: f32, second: f32) -> f32 {
    libm::sqrtf(-2.0 * libm::logf(1.0 - first)) * libm::cosf(TAU as f32 * second)
}
