//! The rand_core traits agree with the inherent API.

use rand_core::{Rng, SeedableRng};
use tandem_rng::Tandem;

#[test]
fn seeds_agree() {
    assert_eq!(Tandem::seed_from_u64(42), Tandem::new(42));
    assert_eq!(Tandem::from_seed(42u128.to_le_bytes()), Tandem::new(42));
}

#[test]
fn words_and_bytes() {
    let mut a = Tandem::new(1);
    let mut b = Tandem::new(1);
    assert_eq!(Rng::next_u32(&mut a), b.next_u32());
    assert_eq!(Rng::next_u64(&mut a), b.next_u64());
    let mut bytes = [0u8; 300];
    let mut want = [0u8; 300];
    a.fill_bytes(&mut bytes);
    b.fill_u8(&mut want);
    assert_eq!(bytes, want);
    assert_eq!(a.position(), b.position());
}

#[test]
fn fork_is_the_spec_fork() {
    let mut a = Tandem::new(3);
    let mut b = Tandem::new(3);
    let child = SeedableRng::fork(&mut a);
    let want = b.fork(1).next().unwrap();
    assert_eq!(child, want);
    assert_eq!(a.position(), b.position());
}

#[test]
fn generic_use() {
    fn mean<R: Rng>(rng: &mut R, n: usize) -> f64 {
        (0..n)
            .map(|_| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
            .sum::<f64>()
            / n as f64
    }
    let m = mean(&mut Tandem::new(9), 100_000);
    assert!((m - 0.5).abs() < 0.01, "{m}");
}
