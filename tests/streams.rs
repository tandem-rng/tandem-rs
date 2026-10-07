//! Long-stream agreement with TandemRNG.jl. The dumps in tests/data are raw little-endian
//! fills written by the Julia reference (tandem-c's tools/dump_streams.jl). Each is compared
//! against a fill, against scalar draws, and at sampled indices against random access.

use std::fs;

use tandem_rng::Tandem;

const KEY: [u32; 4] = [1, 2, 3, 4];

fn dump(name: &str) -> Vec<u8> {
    fs::read(format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"))).expect(name)
}

fn make(name: &str) -> Tandem {
    if name.starts_with("k1234_K8_") {
        Tandem::from_key(KEY, 0, 8)
    } else if name.starts_with("k1234_") {
        Tandem::from_key(KEY, 0, 32)
    } else {
        Tandem::new(42)
    }
}

/// The dump as little-endian values of `N` bytes.
fn values<T, const N: usize>(bytes: &[u8], from: impl Fn([u8; N]) -> T) -> Vec<T> {
    bytes.as_chunks::<N>().0.iter().map(|c| from(*c)).collect()
}

fn check<T: PartialEq + std::fmt::Debug + Copy + Default>(
    name: &str,
    want: &[T],
    fill: impl Fn(&mut Tandem, &mut [T]),
    next: impl Fn(&mut Tandem) -> T,
    at: Option<fn(&Tandem, u64) -> T>,
) {
    let mut a = make(name);
    let mut got = vec![T::default(); want.len()];
    fill(&mut a, &mut got);
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!("{name}: fill differs first at element {i}");
    }
    let mut b = make(name);
    for (i, w) in want.iter().enumerate() {
        assert_eq!(
            next(&mut b),
            *w,
            "{name}: scalar draw differs at element {i}"
        );
    }
    assert_eq!(
        a.position(),
        b.position(),
        "{name}: fill and draws end apart"
    );
    if let Some(at) = at {
        let c = make(name);
        for i in (0..want.len()).step_by(97) {
            assert_eq!(
                at(&c, i as u64),
                want[i],
                "{name}: random access differs at {i}"
            );
        }
    }
}

#[test]
fn u32_k32() {
    let name = "k1234_K32_u32.bin";
    let want = values(&dump(name), u32::from_le_bytes);
    check(
        name,
        &want,
        Tandem::fill_u32,
        Tandem::next_u32,
        Some(Tandem::at_u32),
    );
}

#[test]
fn u64_k32() {
    let name = "k1234_K32_u64.bin";
    let want = values(&dump(name), u64::from_le_bytes);
    check(
        name,
        &want,
        Tandem::fill_u64,
        Tandem::next_u64,
        Some(Tandem::at_u64),
    );
}

#[test]
fn u32_k8() {
    let name = "k1234_K8_u32.bin";
    let want = values(&dump(name), u32::from_le_bytes);
    check(
        name,
        &want,
        Tandem::fill_u32,
        Tandem::next_u32,
        Some(Tandem::at_u32),
    );
}

#[test]
fn f64() {
    let name = "seed42_K32_f64.bin";
    let want = values(&dump(name), f64::from_le_bytes);
    check(
        name,
        &want,
        Tandem::fill_f64,
        Tandem::next_f64,
        Some(Tandem::at_f64),
    );
}

#[test]
fn f32() {
    let name = "seed42_K32_f32.bin";
    let want = values(&dump(name), f32::from_le_bytes);
    check(
        name,
        &want,
        Tandem::fill_f32,
        Tandem::next_f32,
        Some(Tandem::at_f32),
    );
}

#[test]
fn u8() {
    let name = "seed42_K32_u8.bin";
    let want = dump(name);
    check(name, &want, Tandem::fill_u8, Tandem::next_u8, None);
}

#[test]
fn f16_bits() {
    let name = "seed42_K32_f16bits.bin";
    let want = values(&dump(name), u16::from_le_bytes);
    check(
        name,
        &want,
        Tandem::fill_f16_bits,
        Tandem::next_f16_bits,
        None,
    );
}

#[test]
fn char() {
    let name = "seed42_K32_char.bin";
    let want: Vec<char> = values(&dump(name), |b| {
        char::from_u32(u32::from_le_bytes(b)).unwrap()
    });
    check(name, &want, Tandem::fill_char, Tandem::next_char, None);
}

#[test]
fn u128() {
    let name = "seed42_K32_u128.bin";
    let want = values(&dump(name), u128::from_le_bytes);
    check(name, &want, Tandem::fill_u128, Tandem::next_u128, None);
}

#[test]
fn bool() {
    let name = "seed42_K32_bool.bin";
    let want: Vec<bool> = dump(name).iter().map(|&b| b != 0).collect();
    check(name, &want, Tandem::fill_bool, Tandem::next_bool, None);
}

fn pairs<T: Copy>(flat: &[T]) -> Vec<[T; 2]> {
    flat.as_chunks::<2>().0.to_vec()
}

#[test]
fn c64() {
    let name = "seed42_K32_c64.bin";
    let want = pairs(&values(&dump(name), f64::from_le_bytes));
    check(name, &want, Tandem::fill_c64, Tandem::next_c64, None);
}

#[test]
fn c32() {
    let name = "seed42_K32_c32.bin";
    let want = pairs(&values(&dump(name), f32::from_le_bytes));
    check(name, &want, Tandem::fill_c32, Tandem::next_c32, None);
}

#[test]
fn mixed_widths_align() {
    // A u8 then a u64 skips to the next 64-bit boundary, as the spec's alignment rule says.
    let mut rng = Tandem::new(42);
    let _ = rng.next_u8();
    assert_eq!(rng.position(), 8);
    let x = rng.next_u64();
    assert_eq!(rng.position(), 128);
    let mut fresh = Tandem::new(42);
    assert_eq!(fresh.at_u64(1), x);
    fresh.set_position(64);
    assert_eq!(fresh.next_u64(), x);
}

#[test]
fn fills_split_anywhere() {
    // Head, whole rows and tail of a fill agree with one big fill at every offset.
    let mut whole = vec![0u16; 3000];
    Tandem::new(7).fill_u16(&mut whole);
    for start in [0usize, 1, 5, 63, 64, 65, 500, 1024] {
        let mut rng = Tandem::new(7);
        rng.set_position(16 * start as u64);
        let mut part = vec![0u16; 3000 - start];
        rng.fill_u16(&mut part);
        assert_eq!(part, whole[start..], "offset {start}");
    }
}

#[test]
fn cache_history() {
    // Draws of every width and fills in any order, backward jumps included, read what a fresh
    // generator at the same position reads, whatever rows the cache holds or stepped ahead to.
    // K = 1 opens a new group every row.
    for k in [1, 32] {
        let mut rng = Tandem::from_key(KEY, 0, k);
        let mut lcg = 1u64;
        let (mut got, mut want) = ([0u64; 40], [0u64; 40]);
        for step in 0..4000 {
            let mut fresh = Tandem::from_key(KEY, rng.position(), k);
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match lcg >> 61 {
                0 => assert_eq!(rng.next_u8(), fresh.next_u8()),
                1 => assert_eq!(rng.next_u16(), fresh.next_u16()),
                2 => assert_eq!(rng.next_u32(), fresh.next_u32()),
                3 => assert_eq!(rng.next_f32(), fresh.next_f32()),
                4 => assert_eq!(rng.next_f64(), fresh.next_f64()),
                5 => {
                    let n = (lcg >> 32) as usize % 40;
                    rng.fill_u64(&mut got[..n]);
                    fresh.fill_u64(&mut want[..n]);
                    assert_eq!(got[..n], want[..n]);
                }
                6 => {
                    let back = rng.position().min((lcg >> 32) % 3000);
                    rng.set_position(rng.position() - back);
                    fresh.set_position(rng.position());
                }
                _ => assert_eq!(rng.next_u64(), fresh.next_u64()),
            }
            assert_eq!(rng.position(), fresh.position(), "K = {k}, step {step}");
        }
    }
}
