//! A generator survives a serde round trip and continues the same stream.

#![cfg(feature = "serde")]

use tandem_rng::Tandem;

#[test]
fn round_trip_continues_the_stream() {
    let mut a = Tandem::with_chunk_length(42, 8);
    let mut words = [0u32; 77];
    a.fill_u32(&mut words);
    let json = serde_json::to_string(&a).unwrap();
    let mut b: Tandem = serde_json::from_str(&json).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.next_u64(), b.next_u64());
    assert_eq!(a.next_f32(), b.next_f32());
}

#[test]
fn bad_chunk_length_is_an_error() {
    for k in [0, 3, 131072] {
        let json = format!(r#"{{"key":[1,2,3,4],"position":0,"chunk_length":{k}}}"#);
        assert!(serde_json::from_str::<Tandem>(&json).is_err(), "{json}");
    }
}

#[test]
fn round_trip_restores_unsigned_successors() {
    let mut a = Tandem::from_key([1, 2, 3, 4], (1 << 63) - 1, 32);
    a.next_u64();
    assert_eq!(a.position(), (1 << 63) + 64);
    let mut b: Tandem = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
    let (mut want, mut got) = ([0u64; 37], [0u64; 37]);
    a.fill_u64(&mut want);
    b.fill_u64(&mut got);
    assert_eq!(got, want);
    assert_eq!(a, b);
    assert_eq!(a.next_u64(), b.next_u64());
}

fn at(position: u64) -> Tandem {
    serde_json::from_str(&format!(
        r#"{{"key":[1,2,3,4],"position":{position},"chunk_length":32}}"#
    ))
    .unwrap()
}

fn unchanged<T: Copy + PartialEq + std::fmt::Debug>(
    position: u64,
    sentinel: T,
    fill: impl FnOnce(&mut Tandem, &mut [T]),
) {
    unchanged_fill(position, sentinel, 2, fill);
}

fn unchanged_fill<T: Copy + PartialEq + std::fmt::Debug>(
    position: u64,
    sentinel: T,
    len: usize,
    fill: impl FnOnce(&mut Tandem, &mut [T]),
) {
    let mut rng = at(position);
    let original = rng;
    let mut out = vec![sentinel; len];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fill(&mut rng, &mut out);
        }))
        .is_err()
    );
    assert_eq!(rng, original);
    assert_eq!(out, vec![sentinel; len]);
}

#[test]
fn uniform_fill_checks_the_whole_endpoint_before_writing() {
    // Two aligned u64 draws would end exactly at 2^64; a single draw still fits.
    unchanged(u64::MAX - 159, 42u64, Tandem::fill_u64);
    unchanged(u64::MAX - 159, 'x', Tandem::fill_char);
    unchanged(u64::MAX - 159, [42.0; 2], |rng, out| {
        rng.fill_c64(&mut out[..1]);
    });
    unchanged(u64::MAX - 1, true, Tandem::fill_bool);
    let mut rng = at(u64::MAX - 159);
    let mut scalar = rng;
    let mut out = [0u64; 1];
    rng.fill_u64(&mut out);
    assert_eq!(out[0], scalar.next_u64());
    assert_eq!(rng.position(), u64::MAX - 63);

    // Even an empty fill must reject an alignment that is not representable.
    unchanged(u64::MAX, 42u64, |rng, out| rng.fill_u64(&mut out[..0]));
    let mut last = at(u64::MAX - 1);
    last.fill_bool(&mut [false]);
    assert_eq!(last.position(), u64::MAX);
    last.fill_bool(&mut []);
    assert_eq!(last.position(), u64::MAX);
}

#[cfg(feature = "rayon")]
#[test]
fn parallel_fill_checks_before_writing_the_head() {
    unchanged(u64::MAX - 159, 42u64, Tandem::par_fill_u64);
    unchanged(u64::MAX - 159, 42.0f64, Tandem::par_fill_normal_f64);
}

#[test]
fn derived_fills_check_the_complete_draw_sequence() {
    // Each first scratch block fits, but the final draw would end at 2^64.
    unchanged_fill(
        u64::MAX - 513 * 64 + 1,
        42.0f64,
        513,
        Tandem::fill_normal_f64,
    );
    unchanged_fill(
        u64::MAX - 1025 * 64 + 1,
        42.0f64,
        1025,
        Tandem::fill_exponential_f64,
    );
    unchanged_fill(
        u64::MAX - 4098 * 32 + 1,
        42.0f32,
        4097,
        Tandem::fill_normal_f32,
    );
    unchanged(u64::MAX - 63, 42.0f32, |rng, out| {
        rng.fill_normal_f32(&mut out[..1]);
    });
    unchanged(u64::MAX - 63, 42.0f32, Tandem::fill_exponential_f32);
    #[cfg(feature = "alloc")]
    {
        let table = tandem_rng::ChoiceTable::new(&[1.0, 2.0]).unwrap();
        unchanged_fill(u64::MAX - 513 * 64 + 1, 42u32, 513, |rng, out| {
            rng.fill_choice(out, &table);
        });
    }
}
