//! Bounded integers, normals, exponentials and weighted choice against their definitions and
//! their laws. tests/conformance.rs holds the cross-implementation fixtures.

use tandem_rng::Tandem;

/// One bit draw leaves the position unaligned.
fn start() -> Tandem {
    let mut rng = Tandem::new(42);
    rng.next_bool();
    rng
}

#[test]
fn normal_f32_is_box_muller_of_two_f32_draws() {
    // The oracle is the exact formula in f64 on the same two draws. The f32 rounding of
    // the argument, the log and the root adds up to a few ulps against it.
    let (mut a, mut b) = (start(), start());
    for i in 0..1000 {
        let got = a.normal_f32();
        let (u, v) = (f64::from(b.next_f32()), f64::from(b.next_f32()));
        let want = (-2.0 * (1.0 - u).ln()).sqrt() * (std::f64::consts::TAU * v).cos();
        let tol = 16.0 * f64::from(f32::EPSILON) * want.abs().max(1.0);
        assert!(
            (f64::from(got) - want).abs() <= tol,
            "element {i}: {got} against {want}"
        );
    }
    assert_eq!(a, b);
}

/// The bounded fill from its definition: element `i` takes draw `i` of the raw fill, and a
/// rejected draw retries on `sub(purpose).split(g)` of the key at position 0, `g` being the
/// global draw index.
fn below_by_definition(rng: &Tandem, raw: &[u64], n: u64, wide: bool, first: u64) -> Vec<u64> {
    let (bits, purpose) = if wide {
        (64, 0x0042_4c57_3634)
    } else {
        (32, 0x0042_4c57_3332)
    };
    let sub = Tandem::from_key(rng.key(), 0, rng.chunk_length()).sub(purpose);
    let draw = |r: &mut Tandem| {
        if wide {
            r.next_u64()
        } else {
            u64::from(r.next_u32())
        }
    };
    let reject = (1u128 << bits) % u128::from(n.max(1));
    raw.iter()
        .enumerate()
        .map(|(i, &u)| {
            let mut m = u128::from(u) * u128::from(n);
            let low = |m: u128| m & ((1u128 << bits) - 1);
            if low(m) < reject {
                let mut r = sub.split(first + i as u64);
                m = u128::from(draw(&mut r)) * u128::from(n);
                while low(m) < reject {
                    m = u128::from(draw(&mut r)) * u128::from(n);
                }
            }
            (m >> bits) as u64
        })
        .collect()
}

#[test]
fn bounded_fills_follow_the_definition() {
    // Bounds near 2^32 and 2^64 reject a quarter of the draws, so the retry runs.
    for len in [0, 1, 31, 32, 33, 1000, 5000] {
        for n in [0, 1, 3, 1000, 0xc000_0000u32] {
            let mut rng = start();
            let mut raw = start();
            let mut got = vec![0; len];
            rng.fill_below_u32(&mut got, n);
            let mut words = vec![0; len];
            raw.fill_u32(&mut words);
            let words: Vec<u64> = words.iter().map(|&w| u64::from(w)).collect();
            let want = below_by_definition(&rng, &words, u64::from(n), false, 1);
            assert!(
                got.iter().map(|&x| u64::from(x)).eq(want),
                "u32 n={n} len={len}"
            );
            // An empty bounded fill stays at its start, as Appendix A requires.
            let used = if len > 0 { raw } else { start() };
            assert_eq!(rng, used, "u32 consumes exactly len draws");
        }
        for n in [0, 1, 3, 1_000_000_000_000, 0xc000_0000_0000_0000u64] {
            let mut rng = start();
            let mut raw = start();
            let mut got = vec![0; len];
            rng.fill_below_u64(&mut got, n);
            let mut words = vec![0; len];
            raw.fill_u64(&mut words);
            assert_eq!(
                got,
                below_by_definition(&rng, &words, n, true, 1),
                "u64 n={n} len={len}"
            );
            let used = if len > 0 { raw } else { start() };
            assert_eq!(rng, used, "u64 consumes exactly len draws");
        }
    }
}

#[test]
fn bounded_fill_without_rejection_is_the_scalar_draws() {
    // A power of two never rejects.
    let (mut a, mut b) = (start(), start());
    let mut got = vec![0; 777];
    a.fill_below_u32(&mut got, 1 << 20);
    assert!(got.iter().all(|&x| x == b.below_u32(1 << 20)));
    assert_eq!(a, b);
}

#[test]
fn normal_f64_fills_cut_anywhere_equal_the_whole() {
    // The fallback is keyed by the global draw index, so the misses of a fill split at any
    // element equal those of the whole fill. 3000 draws hold about 13 misses. The cuts cross
    // the block of the fill and a row, and the start is unaligned.
    for cut in [0, 1, 15, 16, 777, 1024, 1025, 3000] {
        let mut whole = start();
        let mut a = vec![0.0; 3000];
        whole.fill_normal_f64(&mut a);
        let mut parts = start();
        let mut b = vec![0.0; 3000];
        let (head, tail) = b.split_at_mut(cut);
        parts.fill_normal_f64(head);
        parts.fill_normal_f64(tail);
        assert_eq!(a, b, "cut at {cut}");
        assert_eq!(whole, parts);
    }
}

#[test]
fn normal_f32_fills_are_flattened_pairs() {
    // Lengths cross the block of the normal fill and a row, and include odd ones, which
    // use the cos half of the last pair and still consume both of its draws.
    for len in [0, 1, 2, 127, 128, 129, 255, 257, 300, 1001] {
        let (mut a, mut b) = (Tandem::new(7), Tandem::new(7));
        let mut got = vec![0.0; len];
        a.fill_normal_f32(&mut got);
        let want: Vec<f32> = (0..len.div_ceil(2)).flat_map(|_| b.normal2_f32()).collect();
        assert_eq!(got, want[..len], "normal_f32 at {len}");
        assert_eq!(a, b, "normal_f32 position at {len}");
    }
}

#[test]
fn normals_have_unit_moments() {
    // Mean 0 and variance 1 to within 5 standard errors of 2^20 draws.
    let mut z = vec![0.0; 1 << 20];
    let mut y = vec![0.0f32; 1 << 20];
    Tandem::new(1).fill_normal_f64(&mut z);
    Tandem::new(1).fill_normal_f32(&mut y);
    let y: Vec<f64> = y.iter().map(|&x| f64::from(x)).collect();
    let n = z.len() as f64;
    for (name, z) in [("f64", z), ("f32", y)] {
        let mean = z.iter().sum::<f64>() / n;
        let var = z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 5.0 / n.sqrt(), "{name} mean {mean}");
        assert!(
            (var - 1.0).abs() < 5.0 * (2.0 / n).sqrt(),
            "{name} variance {var}"
        );
    }
}

#[test]
fn exponential_fills_cut_anywhere_equal_the_whole() {
    // Lengths and cuts cross the 1024-element pass of the fill, from an unaligned start. A
    // fill is the scalar draws and ends at the same position. `n = 0` moves nothing.
    for n in [0, 1, 2, 3, 1023, 1024, 1025, 3000] {
        let mut scalar = start();
        let want: Vec<f64> = (0..n).map(|_| scalar.exponential_f64()).collect();
        let mut scalar32 = start();
        let want32: Vec<f32> = (0..n).map(|_| scalar32.exponential_f32()).collect();
        for cut in [0, 1, 7, 1000, 1024, 2049] {
            let cut = cut.min(n);
            let (mut rng, mut got) = (start(), vec![0.0; n]);
            let (head, tail) = got.split_at_mut(cut);
            rng.fill_exponential_f64(head);
            rng.fill_exponential_f64(tail);
            assert_eq!(got, want, "f64 n={n} cut={cut}");
            assert_eq!(rng, scalar, "f64 position n={n} cut={cut}");

            let (mut rng, mut got) = (start(), vec![0.0; n]);
            let (head, tail) = got.split_at_mut(cut);
            rng.fill_exponential_f32(head);
            rng.fill_exponential_f32(tail);
            assert_eq!(got, want32, "f32 n={n} cut={cut}");
            assert_eq!(rng, scalar32, "f32 position n={n} cut={cut}");
        }
    }
    let mut rng = start();
    rng.fill_exponential_f64(&mut []);
    rng.fill_exponential_f32(&mut []);
    assert_eq!(rng, start());
}

/// Moments to the fourth order and a Kolmogorov-Smirnov statistic of the Exp(1) law. The
/// statistic is taken at the edges of 2^16 equal bins of the CDF, which never exceeds the
/// supremum, so the usual critical value is conservative. The samples stream through a small
/// buffer.
fn check_exp1(name: &str, mut fill: impl FnMut(&mut [f64])) {
    const N: usize = 10_000_000;
    const BINS: usize = 1 << 16;
    let (mut sums, mut bins, mut buf) = ([0.0f64; 4], vec![0u32; BINS], vec![0.0; 1 << 14]);
    for _ in 0..N / buf.len() + 1 {
        fill(&mut buf);
        for &x in &buf {
            let cdf = -(-x).exp_m1();
            bins[((cdf * BINS as f64) as usize).min(BINS - 1)] += 1;
            for (k, s) in sums.iter_mut().enumerate() {
                *s += x.powi(k as i32 + 1);
            }
        }
    }
    let n = (N / buf.len() + 1) as f64 * buf.len() as f64;
    // E X^k = k!, and the variance of X^k is (2k)! - (k!)^2.
    for (k, (s, (fact, var))) in sums
        .iter()
        .zip([(1.0, 1.0), (2.0, 20.0), (6.0, 684.0), (24.0, 39744.0)])
        .enumerate()
    {
        let (m, tol) = (s / n, 5.0 * (var / n).sqrt());
        assert!((m - fact).abs() < tol, "{name} moment {}: {m}", k + 1);
    }
    let mut cum = 0.0;
    let d = bins
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            cum += f64::from(b);
            (cum / n - (i + 1) as f64 / BINS as f64).abs()
        })
        .fold(0.0, f64::max);
    // P(sqrt(n) D > 1.95) is 0.001.
    assert!(d * n.sqrt() < 1.95, "{name} KS {}", d * n.sqrt());
}

#[test]
fn exponentials_are_exp1() {
    let mut rng = Tandem::new(1);
    check_exp1("f64", |z| rng.fill_exponential_f64(z));
    let mut rng = Tandem::new(2);
    check_exp1("f32", |z| {
        let mut y = vec![0.0f32; z.len()];
        rng.fill_exponential_f32(&mut y);
        z.iter_mut().zip(y).for_each(|(z, y)| *z = f64::from(y));
    });
}

#[cfg(feature = "alloc")]
#[test]
fn choice_follows_the_weights() {
    // A zero weight never appears. Nine positive weights leave 8 degrees of freedom: the
    // 0.0005 and 0.9995 quantiles of chi-square are 0.71 and 27.87.
    let w = [0.5, 3.0, 0.0, 1.0, 7.0, 2.25, 0.1, 4.0, 1.0, 6.0];
    let table = tandem_rng::ChoiceTable::new(&w).unwrap();
    let mut picks = vec![0; 1_000_000];
    Tandem::new(2028).fill_choice(&mut picks, &table);
    let mut counts = [0f64; 10];
    picks.iter().for_each(|&i| counts[i as usize] += 1.0);
    assert_eq!(counts[2], 0.0);
    let (n, sum) = (picks.len() as f64, w.iter().sum::<f64>());
    let chi2: f64 = (0..w.len())
        .filter(|&i| w[i] > 0.0)
        .map(|i| (counts[i] - n * w[i] / sum).powi(2) / (n * w[i] / sum))
        .sum();
    assert!(0.71 < chi2 && chi2 < 27.87, "chi-square {chi2}");
}
