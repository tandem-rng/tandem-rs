//! Exponential fills as raw little-endian bytes, the same as tandem-c's
//! tools/dump_exponentials.c: `cargo run --release --example dump_exponentials | shasum -a 256`.
//! Every build and target must print the same hash.

use std::io::{BufWriter, Write};

use tandem_rng::Tandem;

const N: usize = 1_000_000;

fn main() {
    let mut out = BufWriter::new(std::io::stdout().lock());
    let mut d = vec![0f64; N];
    let mut f = vec![0f32; N];
    for start in [0u64, 1, 77, 12345, 1 << 30] {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        rng.fill_exponential_f64(&mut d);
        for x in &d {
            out.write_all(&x.to_le_bytes()).unwrap();
        }
        rng.fill_exponential_f32(&mut f);
        for x in &f {
            out.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}
