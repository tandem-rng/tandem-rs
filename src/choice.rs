//! Weighted choice, Appendix C of the specification: Walker's alias table in exact integers,
//! so every port draws the same indices. The table consumes no draws, and each element one
//! `u64` draw with no retry.

use alloc::boxed::Box;
use core::fmt;

use crate::{Tandem, align};

/// Draws per pass of a fill: the draws stay in L1.
const BLOCK: usize = 512;

/// The alias table of Appendix C: index `i` of the weights has probability
/// `q[i] / (m · capacity)`, with the integer masses `q` of the appendix, so within `m · 2^-59`
/// of `w[i] / sum(w)`.
///
/// ```
/// use tandem_rng::{ChoiceTable, Tandem};
///
/// let table = ChoiceTable::new(&[1.0, 2.0, 3.0, 4.0]).expect("valid weights");
/// let mut rng = Tandem::new(42);
/// let i = rng.choice(&table); // in 0..4
/// let mut picks = [0u32; 1000];
/// rng.fill_choice(&mut picks, &table);
/// # let _ = i;
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChoiceTable {
    capacity: u64,
    cut: Box<[u64]>,
    alias: Box<[u32]>,
}

/// The weights of a [`ChoiceTable`] are not 1 to 2^32 − 1 finite, non-negative numbers with
/// a positive one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidWeights;

impl fmt::Display for InvalidWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "weights must number 1 to 2^32 - 1, be finite and not negative, and not all be zero",
        )
    }
}

impl core::error::Error for InvalidWeights {}

/// `w = s · 2^e` with `2^52 ≤ s < 2^53`, for finite `w > 0`.
fn significand(w: f64) -> (u64, i32) {
    let bits = w.to_bits();
    let (exp, mant) = ((bits >> 52) as i32, bits & ((1 << 52) - 1));
    if exp == 0 {
        let shift = mant.leading_zeros() as i32 - 11;
        (mant << shift, -1074 - shift)
    } else {
        (mant | 1 << 52, exp - 1075)
    }
}

/// `ceil(w · 2^t)` for finite `w ≥ 0`, exact. `w · 2^t` in floats could round a positive weight
/// to 0 where the product is subnormal. The caller keeps the result below 2^64.
fn ceil_scaled(w: f64, t: i32) -> u64 {
    if w == 0.0 {
        return 0;
    }
    let (s, e) = significand(w);
    match e + t {
        k if k >= 0 => s << k,
        k if k <= -54 => 1,
        k => (s >> -k) + u64::from(s & ((1 << -k) - 1) != 0),
    }
}

fn nbits(x: u64) -> i32 {
    64 - x.leading_zeros() as i32
}

impl ChoiceTable {
    /// The table of `weights`. `−0.0` counts as zero.
    ///
    /// # Errors
    ///
    /// [`InvalidWeights`] unless `1 ≤ weights.len() < 2^32`, every weight is finite and not
    /// negative, and one is positive.
    pub fn new(weights: &[f64]) -> Result<Self, InvalidWeights> {
        let m = weights.len();
        if m == 0 || m > u32::MAX as usize {
            return Err(InvalidWeights);
        }
        let mut wmax = 0.0f64;
        for &w in weights {
            if !(w.is_finite() && w >= 0.0) {
                return Err(InvalidWeights);
            }
            wmax = wmax.max(w);
        }
        if wmax == 0.0 {
            return Err(InvalidWeights);
        }
        // A first pass at a scale that cannot overflow bounds the sum, and the second scales
        // it to just below 2^63. `significand(wmax).1 + 52` is the exponent of `wmax`.
        let m64 = m as u64;
        let mut t = 63 - nbits(m64) - (significand(wmax).1 + 52);
        let first: u64 = weights.iter().map(|&w| ceil_scaled(w, t)).sum();
        t += 63 - nbits(first);
        let mut cut: Box<[u64]> = weights.iter().map(|&w| ceil_scaled(w, t)).collect();
        let total: u64 = cut.iter().sum();
        let capacity = total.div_ceil(m64);
        // The padding goes to the first of the largest masses.
        let big = (0..m).fold(0, |b, i| if cut[i] > cut[b] { i } else { b });
        cut[big] += capacity * m64 - total;

        // Pair the columns in place: `cut` holds a column's mass until it is paired.
        let mut alias: Box<[u32]> = (0..m as u32).collect();
        let mut l = cut
            .iter()
            .position(|&c| c >= capacity)
            .expect("a full column");
        for i in 0..m {
            let mut j = i;
            while j <= i && cut[j] < capacity {
                alias[j] = l as u32;
                cut[l] -= capacity - cut[j];
                j = l;
                if cut[l] < capacity {
                    l += 1;
                    while l < m && cut[l] < capacity {
                        l += 1;
                    }
                }
            }
        }
        Ok(ChoiceTable {
            capacity,
            cut,
            alias,
        })
    }

    /// The mass `S` of one column.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The share of column `j` that stays with index `j`, out of [`capacity`](Self::capacity).
    /// It has one entry per weight.
    pub fn cut(&self) -> &[u64] {
        &self.cut
    }

    /// The index that takes the rest of column `j`.
    pub fn alias(&self) -> &[u32] {
        &self.alias
    }

    /// The index of the `u64` draw `r`.
    #[inline]
    pub(crate) fn index(&self, r: u64) -> u32 {
        let x = u128::from(r) * self.cut.len() as u128;
        let j = (x >> 64) as usize;
        let v = (u128::from(x as u64) * u128::from(self.capacity)) >> 64;
        if (v as u64) < self.cut[j] {
            j as u32
        } else {
            self.alias[j]
        }
    }
}

impl Tandem {
    /// An index into the weights of `table`, from one `u64` draw. It equals element 0 of
    /// [`fill_choice`](Self::fill_choice).
    #[inline]
    pub fn choice(&mut self, table: &ChoiceTable) -> u32 {
        table.index(self.next_u64())
    }

    /// Fill with indices into the weights of `table`. Element `i` comes from draw `i` of the
    /// `u64` fill, with no retry, so the fill equals the scalar draws and a fill cut anywhere
    /// equals the whole. An empty fill aligns the position to 64 bits.
    pub fn fill_choice(&mut self, out: &mut [u32], table: &ChoiceTable) {
        self.pos = align(self.pos, 64);
        let mut draws = [0u64; BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            let draws = &mut draws[..chunk.len()];
            self.fill_u64(draws);
            for (o, &r) in chunk.iter_mut().zip(draws.iter()) {
                *o = table.index(r);
            }
        }
    }
}
