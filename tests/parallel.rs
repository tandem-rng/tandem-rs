//! Parallel fills equal the serial fills in values and final position.

#![cfg(feature = "rayon")]

use tandem_rng::Tandem;

/// Start offsets and lengths that cut rows, tasks and chunks: one task is 1024 rows. The
/// empty fill starts unaligned.
const CASES: [(u64, usize); 7] = [
    (33, 0),
    (0, 5),
    (96, 300),
    (1024, 1 << 15),
    (70, (1 << 15) + 33),
    (0, 300_000),
    (12_345, 600_001),
];

macro_rules! agrees {
    ($name:ident, $ty:ty, $fill:ident, $par:ident, $k:expr) => {
        #[test]
        fn $name() {
            for (pos, len) in CASES {
                let mut serial = Tandem::from_key([1, 2, 3, 4], pos, $k);
                let mut par = serial;
                let mut want = vec![<$ty>::default(); len];
                let mut got = vec![<$ty>::default(); len];
                serial.$fill(&mut want);
                par.$par(&mut got);
                assert!(got == want, "values at position {pos}, length {len}");
                assert_eq!(par, serial, "position at {pos}, length {len}");
                assert_eq!(par.next_u32(), serial.next_u32(), "the stream continues");
            }
        }
    };
}

agrees!(u32_k32, u32, fill_u32, par_fill_u32, 32);
agrees!(u32_k1, u32, fill_u32, par_fill_u32, 1);
agrees!(u64_k32, u64, fill_u64, par_fill_u64, 32);
agrees!(f32_k8, f32, fill_f32, par_fill_f32, 8);
agrees!(f64_k32, f64, fill_f64, par_fill_f64, 32);

#[test]
fn large_groups_keep_the_serial_stream() {
    const GROUP: usize = 65536 * 32;
    for (pos, len) in [
        (33, 0),
        (33, 5),
        (0, GROUP),
        (32, GROUP + 17),
        ((65536 - 17) * 1024 + 13, 2 * GROUP + 49),
    ] {
        let mut serial = Tandem::from_key([1, 2, 3, 4], pos, 65536);
        let mut par = serial;
        let (mut want, mut got) = (vec![0u32; len], vec![0u32; len]);
        serial.fill_u32(&mut want);
        par.par_fill_u32(&mut got);
        assert_eq!(got, want, "position {pos}, length {len}");
        assert_eq!(par, serial);
        assert_eq!(par.next_u64(), serial.next_u64());
    }
}

#[test]
fn below_fills() {
    for (pos, len) in CASES {
        let mut serial = Tandem::from_key([1, 2, 3, 4], pos, 32);
        let mut par = serial;
        let (mut want, mut got) = (vec![0u32; len], vec![0u32; len]);
        serial.fill_below_u32(&mut want, 0xc000_0000);
        par.par_fill_below_u32(&mut got, 0xc000_0000);
        assert!(got == want, "u32 at position {pos}, length {len}");
        assert_eq!(par, serial);
        let (mut want, mut got) = (vec![0u64; len], vec![0u64; len]);
        serial.fill_below_u64(&mut want, 0xc000_0000_0000_0000);
        par.par_fill_below_u64(&mut got, 0xc000_0000_0000_0000);
        assert!(got == want, "u64 at position {pos}, length {len}");
        assert_eq!(par, serial);
    }
}

#[test]
fn by_draw_fills() {
    // Normals and choices. One task is 8192 elements; the odd lengths end on a half pair, and
    // the empty fills start unaligned.
    for (pos, len) in [
        (33, 0),
        (70, 5),
        (0, 8192),
        (96, 8193),
        (1, 40_001),
        (0, 100_000),
    ] {
        let mut serial = Tandem::from_key([1, 2, 3, 4], pos, 32);
        let mut par = serial;
        let (mut want, mut got) = (vec![0f64; len], vec![0f64; len]);
        serial.fill_normal_f64(&mut want);
        par.par_fill_normal_f64(&mut got);
        assert!(got == want, "f64 at position {pos}, length {len}");
        assert_eq!(par, serial);
        let (mut want, mut got) = (vec![0f32; len], vec![0f32; len]);
        serial.fill_normal_f32(&mut want);
        par.par_fill_normal_f32(&mut got);
        assert!(got == want, "f32 at position {pos}, length {len}");
        assert_eq!(par, serial);
        let table = tandem_rng::ChoiceTable::new(&[1.0, 2.0, 0.0, 4.0]).unwrap();
        let (mut want, mut got) = (vec![0u32; len], vec![0u32; len]);
        serial.fill_choice(&mut want, &table);
        par.par_fill_choice(&mut got, &table);
        assert!(got == want, "choice at position {pos}, length {len}");
        assert_eq!(par, serial);
    }
}

#[test]
fn large_group_draws_preserve_pairs_and_partial_groups() {
    let table = tandem_rng::ChoiceTable::new(&[1.0, 2.0, 0.0, 4.0]).unwrap();
    // Odd/even first f32 draw, alignment onto a group boundary, and a cached late start.
    for pos in [1, 33, 1024 * 1024 - 1, 1024 * 1017 + 13] {
        let mut serial = Tandem::from_key([1, 2, 3, 4], pos, 1024);
        let mut par = serial;
        let len = 2 * 1024 * 32 + 17;
        let (mut want, mut got) = (vec![0f32; len], vec![0f32; len]);
        serial.fill_normal_f32(&mut want);
        par.par_fill_normal_f32(&mut got);
        assert_eq!(got, want, "normal32 at {pos}");
        assert_eq!(par, serial);
        let (mut want, mut got) = (vec![0f64; len], vec![0f64; len]);
        serial.fill_normal_f64(&mut want);
        par.par_fill_normal_f64(&mut got);
        assert_eq!(got, want, "normal64 at {pos}");
        assert_eq!(par, serial);
        let (mut want, mut got) = (vec![0u32; len], vec![0u32; len]);
        serial.fill_choice(&mut want, &table);
        par.par_fill_choice(&mut got, &table);
        assert_eq!(got, want, "choice at {pos}");
        assert_eq!(par, serial);
        assert_eq!(par.next_u64(), serial.next_u64());
    }
}
