//! `f64` normals: the 1024-layer ziggurat of Appendix A, one `u64` draw per element.
//!
//! A draw that misses the inner rectangles continues on its own fallback generator,
//! `from_key(key, 0, K).sub(PURPOSE_NORMAL64).split(g)`, where `g` is the global index of the
//! draw. So element `i` depends on draw `i` alone, and a fill cut anywhere equals the whole.

use crate::boxmuller::neg2_log;
use crate::zig_tables::{K, R, W, Y};
use crate::{AUX_STREAM, DOMAIN_SPLIT, DOMAIN_STREAM, Tandem, align, block, f_lanes, to_f64};

/// Reserved purpose of the fallback generators.
const PURPOSE_NORMAL64: u64 = 0x4e_524d_3634;

/// Draws per pass of a fill: the draws and their tables stay in L1.
const BLOCK: usize = 512;

/// `(±ra W[i], ra < K[i])`: the candidate of a draw and whether it lies in the inner rectangle.
/// `W` holds `-W[i]` at `1024 + i`, so bit 10 picks the sign by the index.
#[inline(always)]
fn candidate(r: u64) -> (f64, bool) {
    let ra = r >> 11;
    // `ra < 2^53` converts exactly, and the signed conversion is the faster one on x86_64.
    let x = (ra as i64) as f64 * W[(r & 2047) as usize];
    (x, ra < K[(r & 1023) as usize])
}

/// The `u64` draws of a fallback generator from its position 0. Draw `d` sits in row `d / 16`
/// and lane `(d / 2) mod 8`, so each pair of draws costs one `block` instead of a row of eight.
struct Fallback {
    key: [u32; 4],
    k: u64,
    d: u32,
    blk: [u32; 4],
}

impl Fallback {
    /// The fallback of draw `g` by the public `split`, for the scalar draw.
    fn new(sub: &Tandem, g: u64) -> Fallback {
        let key = sub.split(g).key;
        Fallback {
            key,
            k: u64::from(sub.k),
            d: 0,
            blk: block(&key, 0, 0),
        }
    }

    /// The fallbacks of eight draws, with `split` and the first block run on eight lanes at
    /// once. That costs about a quarter of eight separate seedings. Equal scalar draws and fills
    /// check it against [`new`](Self::new).
    fn eight(sub: &Tandem, g: [u64; 8]) -> [Fallback; 8] {
        let mut o = [
            g.map(|g| (g >> 1) as u32),
            g.map(|g| (g >> 33) as u32),
            [DOMAIN_SPLIT; 8],
            [0; 8],
        ];
        let mut h = sub.key.map(|w| [w; 8]);
        f_lanes(&mut o, &mut h, 0);
        // `split` keeps the hidden half for odd indices.
        let keys: [[u32; 8]; 4] =
            core::array::from_fn(|w| core::array::from_fn(|l| [o, h][(g[l] & 1) as usize][w][l]));
        let mut o = [[0; 8], [0; 8], [DOMAIN_STREAM; 8], [AUX_STREAM; 8]];
        let mut h = keys;
        f_lanes(&mut o, &mut h, 1);
        core::array::from_fn(|l| Fallback {
            key: keys.map(|w| w[l]),
            k: u64::from(sub.k),
            d: 0,
            blk: o.map(|w| w[l]),
        })
    }

    fn next(&mut self) -> u64 {
        let d = self.d;
        self.d += 1;
        if d >= 2 && d & 1 == 0 {
            let row = u64::from(d >> 4);
            let chunk = 8 * (row / self.k) + u64::from((d >> 1) & 7);
            self.blk = block(&self.key, chunk, (row % self.k) as u32);
        }
        let i = 2 * (d & 1) as usize;
        u64::from(self.blk[i]) | u64::from(self.blk[i + 1]) << 32
    }
}

/// The slow path of a draw `r` that missed, on the fallback draws `next`. `ln` is
/// `-0.5 neg2_log`, exact given `neg2_log`. Rust does not contract, so every other operation
/// rounds once, as the appendix requires.
#[cold]
#[inline(never)]
fn slow(mut r: u64, mut next: impl FnMut() -> u64) -> f64 {
    loop {
        let i = (r & 1023) as usize;
        let (x, inner) = candidate(r);
        if inner {
            return x;
        }
        if i == 0 {
            // The tail beyond R, by Marsaglia's method.
            loop {
                let a = 0.5 * neg2_log(1.0 - to_f64(next())) / R;
                let b = 0.5 * neg2_log(1.0 - to_f64(next()));
                if b + b >= a * a {
                    return if (r >> 10) & 1 == 1 { -(R + a) } else { R + a };
                }
            }
        }
        let y = Y[i] + to_f64(next()) * (Y[i + 1] - Y[i]);
        if -0.5 * neg2_log(y) < -0.5 * (x * x) {
            return x;
        }
        r = next();
    }
}

/// The ziggurat on a generator without a key: a miss continues on the same stream. The values
/// are standard normals, but not the Tandem stream's.
#[cfg(feature = "rand")]
#[inline]
pub(crate) fn normal_f64_on(mut next: impl FnMut() -> u64) -> f64 {
    let r = next();
    match candidate(r) {
        (x, true) => x,
        _ => slow(r, next),
    }
}

/// Resolve the misses `(index in out, draw)`, whose global draw indices are `first + index`,
/// eight at a time. A short last group repeats its last index and leaves the spare lanes unused.
fn resolve(out: &mut [f64], misses: &[(usize, u64)], first: u64, sub: &Tandem) {
    for group in misses.chunks(8) {
        let g = core::array::from_fn(|l| first + group[l.min(group.len() - 1)].0 as u64);
        for (&(i, r), f) in group.iter().zip(&mut Fallback::eight(sub, g)) {
            out[i] = slow(r, || f.next());
        }
    }
}

impl Tandem {
    /// A standard normal from one `u64` draw by the ziggurat of Appendix A. It equals element 0
    /// of [`fill_normal_f64`](Self::fill_normal_f64), and with `std` it matches tandem-c bit for
    /// bit.
    #[inline]
    pub fn normal_f64(&mut self) -> f64 {
        let g = align(self.pos, 64) / 64;
        let r = self.next_u64();
        match candidate(r) {
            (x, true) => x,
            _ => self.normal_f64_miss(r, g),
        }
    }

    // Seeding the fallback is long, so it stays out of the inlined fast path.
    #[cold]
    #[inline(never)]
    fn normal_f64_miss(&self, r: u64, g: u64) -> f64 {
        let mut f = Fallback::new(&self.normal_sub(), g);
        slow(r, || f.next())
    }

    /// Fill with standard normals. Element `i` comes from draw `i` of the `u64` fill and the
    /// fill consumes `out.len()` draws. An empty fill aligns the position to 64 bits.
    pub fn fill_normal_f64(&mut self, out: &mut [f64]) {
        let first = align(self.pos, 64) / 64;
        self.pos = 64 * first;
        let mut sub = None;
        let mut draws = [0u64; BLOCK];
        // Misses queue across passes, so that their fallbacks are seeded eight at a time.
        let mut misses = [(0, 0); BLOCK + 7];
        let (mut nm, n) = (0, out.len());
        for start in (0..n).step_by(BLOCK) {
            let chunk = &mut out[start..n.min(start + BLOCK)];
            let draws = &mut draws[..chunk.len()];
            self.fill_u64(draws);
            for (j, (z, &r)) in chunk.iter_mut().zip(draws.iter()).enumerate() {
                let (x, inner) = candidate(r);
                *z = x;
                if !inner {
                    misses[nm] = (start + j, r);
                    nm += 1;
                }
            }
            let full = nm / 8 * 8;
            if full > 0 {
                let sub = sub.get_or_insert_with(|| self.normal_sub());
                resolve(out, &misses[..full], first, sub);
                misses.copy_within(full..nm, 0);
                nm -= full;
            }
        }
        if nm > 0 {
            let sub = sub.get_or_insert_with(|| self.normal_sub());
            resolve(out, &misses[..nm], first, sub);
        }
    }

    /// The parent of the fallback generators.
    fn normal_sub(&self) -> Tandem {
        Tandem::from_key(self.key, 0, self.k).sub(PURPOSE_NORMAL64)
    }
}
