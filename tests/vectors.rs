//! Every vector of the specification, from tests/vectors_data.rs.

mod vectors_data;

use tandem_rng::{Tandem, block, f_keyed, t};
use vectors_data::*;

const DOMAIN_STREAM: u32 = 0x9e37_79b9;
const AUX_STREAM: u32 = 0x94d0_49bb;

#[test]
fn step() {
    for &(o, h, o_out, h_out) in T {
        let (mut o, mut h) = (o, h);
        t(&mut o, &mut h);
        assert_eq!((o, h), (o_out, h_out));
    }
}

#[test]
fn seeding_function() {
    for &(counter, o, h) in F {
        assert_eq!(f_keyed(&KEY, counter, DOMAIN_STREAM, AUX_STREAM), (o, h));
    }
}

#[test]
fn stream_words() {
    let rng = Tandem::from_key(KEY, 0, K);
    for &(first, words) in STREAM {
        for (i, &w) in words.iter().enumerate() {
            assert_eq!(rng.at_u32(first + i as u64), w);
        }
        let row = (first * 32) >> 10;
        let lane = ((first * 32) >> 7) & 7;
        assert_eq!(
            block(
                &KEY,
                8 * (row / u64::from(K)) + lane,
                (row % u64::from(K)) as u32
            ),
            words
        );
    }
    let mut filled = vec![0u32; 64];
    Tandem::from_key(KEY, 0, K).fill_u32(&mut filled);
    for &(first, words) in STREAM {
        assert_eq!(&filled[first as usize..first as usize + 4], &words);
    }
}

#[test]
fn draws_from_position_0() {
    let rng = Tandem::from_key(KEY, 0, K);
    for &(i, x) in F64 {
        assert_eq!(rng.at_f64(i), x);
    }
    for &(i, x) in F32 {
        assert_eq!(rng.at_f32(i), x);
    }
    let mut bits = vec![false; 129];
    Tandem::from_key(KEY, 0, K).fill_bool(&mut bits);
    for &(i, b) in BOOL {
        assert_eq!(bits[i as usize], b);
    }
}

#[test]
fn derived_keys() {
    let mut rng = Tandem::from_key(KEY, 0, K);
    assert_eq!(rng.split(0).key(), SPLIT0);
    assert_eq!(rng.split(1).key(), SPLIT1);
    assert_eq!(rng.sub(7).key(), PURPOSE7);
    let kids: Vec<Tandem> = rng.fork(2).collect();
    assert_eq!(kids[0].key(), FORK0);
    assert_eq!(kids[0].position(), 0);
    assert_eq!(kids[0].chunk_length(), K);
    assert_eq!(rng.position(), 128);
}

#[test]
fn seed_whitening() {
    let rng = Tandem::with_chunk_length(SEED, K);
    assert_eq!(rng.key(), SEED_KEY);
    assert_eq!(Tandem::new(SEED), rng);
    for &(i, x) in SEED_F64 {
        assert_eq!(rng.at_f64(i), x);
    }
    for &(i, x) in SEED_U32 {
        assert_eq!(rng.at_u32(i), x);
    }
}
