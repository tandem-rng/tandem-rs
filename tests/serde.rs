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
