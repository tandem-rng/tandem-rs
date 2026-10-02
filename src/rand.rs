//! `rand_core` integration.
//!
//! `SeedableRng::from_seed` treats the 16 seed bytes as a little-endian 128-bit integer seed
//! and whitens it as the specification requires, so `Tandem::seed_from_u64(42)` is the same
//! generator as `Tandem::new(42)` and as `Tandem8x32(42)` in Julia. `SeedableRng::fork` is
//! the specification's fork of one child.

use core::convert::Infallible;

use rand_core::{Rng, SeedableRng, TryRng};

use crate::Tandem;

impl TryRng for Tandem {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.next_u32())
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.next_u64())
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        self.fill_u8(dst);
        Ok(())
    }
}

impl SeedableRng for Tandem {
    type Seed = [u8; 16];

    fn from_seed(seed: [u8; 16]) -> Self {
        Tandem::new(u128::from_le_bytes(seed))
    }

    fn seed_from_u64(state: u64) -> Self {
        Tandem::new(u128::from(state))
    }

    fn fork(&mut self) -> Self
    where
        Self: Rng,
    {
        Tandem::fork(self, 1).next().expect("one child")
    }
}
