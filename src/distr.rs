//! `rand` distributions of the derived draws of Appendix A.
//!
//! Each `sample` equals the inherent scalar draw bit for bit, so `sample_iter` equals the
//! repeated scalar draws. Bounded integers, `f32` normals and exponentials need only the `u32`
//! and `u64` draws, so they are exact on every generator that yields the Tandem stream. An `f64`
//! normal that misses the ziggurat's fast path continues on a fallback keyed by the generator's
//! key and the draw's index, so `sample` needs the [`Tandem`] itself: it takes the exact path
//! when the generator is a `Tandem` or a `&mut Tandem`. Any other generator, a `dyn Rng` or a
//! wrapper of a `Tandem` among them, gets the same ziggurat with its misses continued on its own
//! stream. Those values are standard normals, but they leave the Tandem stream at the first
//! miss.

use ::rand::Rng;
use ::rand::distr::Distribution;

use crate::boxmuller::{exponential_f32, exponential_f64, pair_f32};
use crate::derived::{below_u32, below_u64};
use crate::ziggurat::normal_f64_on;
use crate::{Tandem, to_f32, to_f64};

/// Standard normals. `f64` is [`Tandem::normal_f64`], `f32` is [`Tandem::normal_f32`] and
/// `[f32; 2]` is [`Tandem::normal2_f32`], so the flattened pairs equal
/// [`Tandem::fill_normal_f32`].
///
/// ```
/// use rand::{RngExt, distr::Distribution};
/// use tandem_rng::{StandardNormal, Tandem};
///
/// let mut rng = Tandem::new(42);
/// let z: f64 = rng.sample(StandardNormal);
/// let zs: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(4).collect();
/// # let _ = (z, zs);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StandardNormal;

/// Exponentials of rate 1. `f64` is [`Tandem::exponential_f64`] and `f32` is
/// [`Tandem::exponential_f32`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Exp1;

/// Uniform integers in `0..n`. `Below(n)` with a `u32` is [`Tandem::below_u32`] and with a
/// `u64` is [`Tandem::below_u64`], so a bound of 0 returns 0 after one draw. A fill
/// ([`Tandem::fill_below_u32`]) retries a rejected draw on a fallback instead of the stream,
/// so `sample_iter` equals it only up to the first rejection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Below<T>(pub T);

impl Distribution<f64> for StandardNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match tandem(rng) {
            Some(t) => t.normal_f64(),
            None => normal_f64_on(|| rng.next_u64()),
        }
    }
}

impl Distribution<f32> for StandardNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        let [z, _]: [f32; 2] = self.sample(rng);
        z
    }
}

impl Distribution<[f32; 2]> for StandardNormal {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f32; 2] {
        let a = to_f32(rng.next_u32());
        let b = to_f32(rng.next_u32());
        pair_f32(a, b)
    }
}

impl Distribution<f64> for Exp1 {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        exponential_f64(to_f64(rng.next_u64()))
    }
}

impl Distribution<f32> for Exp1 {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        exponential_f32(to_f32(rng.next_u32()))
    }
}

impl Distribution<u32> for Below<u32> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        below_u32(self.0, || rng.next_u32())
    }
}

/// [`Tandem::choice`]: one `u64` draw, so exact on any generator of the Tandem stream.
#[cfg(feature = "alloc")]
impl Distribution<u32> for crate::ChoiceTable {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        self.index(rng.next_u64())
    }
}

impl Distribution<u64> for Below<u64> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        below_u64(self.0, || rng.next_u64())
    }
}

/// `rng` as a `Tandem` when `R` is `Tandem` or `&mut Tandem`. `Distribution::sample` cannot
/// bound `R`, and `Any` needs `'static`, so the check compares type ids with lifetimes erased.
/// It folds to a constant per `R`.
#[allow(unsafe_code)]
#[inline(always)]
fn tandem<R: ?Sized>(rng: &mut R) -> Option<&mut Tandem> {
    let id = typeid::of::<R>();
    if id == typeid::of::<Tandem>() {
        // SAFETY: the ids agree up to lifetimes and `Tandem` has none, so `R` is `Tandem`.
        Some(unsafe { &mut *(rng as *mut R).cast::<Tandem>() })
    } else if id == typeid::of::<&mut Tandem>() {
        // SAFETY: `R` is `&'a mut Tandem` for an `'a` that outlives the borrow of `rng`, and
        // the result lives no longer than that borrow.
        Some(unsafe { &mut **(rng as *mut R).cast::<&mut Tandem>() })
    } else {
        None
    }
}
