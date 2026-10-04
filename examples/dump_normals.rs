//! `f64` normal fills as raw little-endian bytes, the same as tandem-c's tools/dump_normals.c:
//! `cargo run --release --example dump_normals | shasum -a 256`. Every build and target must
//! print the same hash.

use std::io::Write;

use tandem_rng::Tandem;

fn main() {
    let mut out = std::io::stdout().lock();
    let mut d = vec![0f64; 1_000_000];
    for start in [0u64, 1, 77, 12345, 1 << 30] {
        let mut rng = Tandem::new(2026 | 7 << 64);
        rng.set_position(start);
        rng.fill_normal_f64(&mut d);
        let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
        out.write_all(&bytes).unwrap();
    }
}
