//! Normal fills as raw little-endian bytes, the same as tandem-c's tools/dump_normals.c:
//! `cargo run --release --example dump_normals | shasum -a 256`. Every build and target must
//! print the same hash.

use std::io::Write;

use tandem_rng::Tandem;

const PAIRS: usize = 1_000_000;

fn main() {
    let mut out = std::io::stdout().lock();
    let mut d = vec![0f64; 2 * PAIRS - 1];
    let mut f = vec![0f32; 2 * PAIRS - 1];
    for start in [0u64, 1, 77, 12345, 1 << 30] {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        // An odd count drops the last sin half.
        rng.fill_normal_f64(&mut d);
        out.write_all(&bytes(&d)).unwrap();
        rng.fill_normal_f32(&mut f);
        out.write_all(&bytes(&f)).unwrap();
    }
}

/// Little-endian bytes of the values, as the C tool writes them on its little-endian hosts.
trait Le: Copy {
    fn push(self, out: &mut Vec<u8>);
}

impl Le for f64 {
    fn push(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl Le for f32 {
    fn push(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

fn bytes<T: Le>(x: &[T]) -> Vec<u8> {
    let mut v = Vec::with_capacity(x.len() * 8);
    for &e in x {
        e.push(&mut v);
    }
    v
}
