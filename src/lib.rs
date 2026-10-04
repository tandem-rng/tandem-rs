//! Tandem8x32: a noncryptographic pseudorandom number generator, fast on CPUs and GPUs alike.
//!
//! This crate implements the [specification](https://github.com/tandem-rng/spec) and
//! produces the stream the specification defines, bit for bit. A [`Tandem`] is its
//! transport form (128-bit key, 64-bit bit position, chunk length `K`) plus a cache of the
//! current 1024-bit row, so it is `Copy` and cheap to clone.
//!
//! ```
//! use tandem_rng::Tandem;
//!
//! let mut rng = Tandem::new(42);
//! let x: f64 = rng.next_f64();
//! let mut words = [0u32; 1024];
//! rng.fill_u32(&mut words);
//! let worker = rng.split(7);               // by index, from the key alone
//! let kids: Vec<Tandem> = rng.fork(4).collect(); // from the current block, parent moves on
//! # let _ = (x, worker, kids);
//! ```
//!
//! `Tandem` implements `rand_core`'s [`TryRng`](rand_core::TryRng) (and so
//! [`Rng`](rand_core::Rng)) and [`SeedableRng`](rand_core::SeedableRng), so it drives every
//! `rand` distribution.
//!
//! The `wgpu` feature adds [`gpu::GpuFill`], the same fill run as a compute shader.
//!
//! The `rayon` feature adds `par_fill_u32`, `par_fill_u64`, `par_fill_f32` and `par_fill_f64`:
//! the same fills, split at row boundaries across threads.
//!
//! [`Tandem::below_u32`], [`Tandem::normal_f64`] and their fills are not part of the
//! specification. They follow the shared device core so every port agrees.
//!
//! The default `std` feature adds fused multiply-adds and the square root of the normals from
//! the standard library. Without it the crate is `no_std`.
//!
//! The `serde` feature serializes a [`Tandem`] as its transport form: key, position and chunk
//! length.
//!
//! The `simd-intrinsics` feature spells the widening multiply and the float row stores with
//! NEON or SSE2 intrinsics. It admits `unsafe` in one private module and leaves the stream
//! as it is.

#![no_std]
// The `simd-intrinsics` feature admits `unsafe` in `arch` alone.
#![cfg_attr(not(feature = "simd-intrinsics"), forbid(unsafe_code))]
#![cfg_attr(feature = "simd-intrinsics", deny(unsafe_code))]
#![warn(missing_docs)]

#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(feature = "simd-intrinsics")]
mod arch;
mod boxmuller;
mod derived;
#[cfg(feature = "wgpu")]
pub mod gpu;
#[cfg(feature = "rayon")]
mod parallel;
mod rand;
#[cfg(feature = "serde")]
mod transport;

use wide::{u32x4, u64x2};

/// The default chunk length `K`.
pub const DEFAULT_K: u32 = 32;

const CLOCK_WEYL: u32 = 0x9e37_79b9;
const DOMAIN_STREAM: u32 = 0x9e37_79b9;
const DOMAIN_SPLIT: u32 = 0xbb67_ae85;
const DOMAIN_FORK: u32 = 0xd251_1f53;
const DOMAIN_FOLD: u32 = 0xcd9e_8d57;
const DOMAIN_SEED: u32 = 0xa54f_f53a;
const AUX_STREAM: u32 = 0x94d0_49bb;
const RC: [u32; 8] = [
    0xd17c_c1b7,
    0xa722_0a94,
    0xfe13_abe8,
    0xfa9a_6ee0,
    0xedb1_4acc,
    0x9e21_c820,
    0xff28_b1d5,
    0xef5d_e2b0,
];

/// The step `T`: mix the exposed half, clock the hidden half, feed `o0` back into `h0`.
#[inline]
pub fn t(o: &mut [u32; 4], h: &mut [u32; 4]) {
    let p0 = u64::from(o[0]) * u64::from(h[0] | 1);
    let p1 = u64::from(o[2]) * u64::from(h[1] | 1);
    let (lo0, hi0) = (p0 as u32, (p0 >> 32) as u32);
    let (lo1, hi1) = (p1 as u32, (p1 >> 32) as u32);
    let n = [
        o[1] ^ hi1 ^ lo1,
        lo1.rotate_left(16) ^ h[2],
        o[3] ^ hi0 ^ lo0,
        lo0.rotate_left(16) ^ h[3],
    ];
    h[0] ^= h[1].rotate_left(7);
    h[1] ^= h[2].rotate_left(13);
    h[2] ^= h[3].rotate_left(22);
    h[3] ^= h[0].rotate_left(3);
    h[0] = h[0].wrapping_add(CLOCK_WEYL) ^ n[0];
    *o = n;
}

/// The seeding function `F`: eight rounds of `T`, a round constant, and a half swap.
pub fn f(o: &mut [u32; 4], h: &mut [u32; 4]) {
    for rc in RC {
        t(o, h);
        o[0] ^= rc;
        core::mem::swap(o, h);
    }
}

/// `F(key, counter, domain, aux)`: the state `F` produces from a keyed input block.
pub fn f_keyed(key: &[u32; 4], counter: u64, domain: u32, aux: u32) -> ([u32; 4], [u32; 4]) {
    let mut o = [counter as u32, (counter >> 32) as u32, domain, aux];
    let mut h = *key;
    f(&mut o, &mut h);
    (o, h)
}

/// Block `B(c, j)`: the exposed half of chunk `c` after `j + 1` steps.
pub fn block(key: &[u32; 4], c: u64, j: u32) -> [u32; 4] {
    let (mut o, mut h) = f_keyed(key, c, DOMAIN_STREAM, AUX_STREAM);
    for _ in 0..=j {
        t(&mut o, &mut h);
    }
    o
}

// ---- Eight lanes --------------------------------------------------------------------------
//
// A row is the eight chunks of a group at one step. The state is stored word-major, one
// four-lane vector per word and two quads per row, so every operation of T is one vector
// instruction and the row store is a 4x4 transpose. `wide` lowers this to NEON, SSE or
// scalar code as the target allows.

#[derive(Clone, Copy)]
struct Quad {
    o: [u32x4; 4],
    h: [u32x4; 4],
}

#[derive(Clone, Copy)]
struct Lanes {
    q: [Quad; 2],
}

// Hoisted: materialising a splat inside the step costs a call per use.
const WEYL4: u32x4 = u32x4::splat(CLOCK_WEYL);
const ONE4: u32x4 = u32x4::splat(1);
const DOMAIN_STREAM4: u32x4 = u32x4::splat(DOMAIN_STREAM);
const AUX_STREAM4: u32x4 = u32x4::splat(AUX_STREAM);
const RC4: [u32x4; 8] = [
    u32x4::splat(RC[0]),
    u32x4::splat(RC[1]),
    u32x4::splat(RC[2]),
    u32x4::splat(RC[3]),
    u32x4::splat(RC[4]),
    u32x4::splat(RC[5]),
    u32x4::splat(RC[6]),
    u32x4::splat(RC[7]),
];

#[inline(always)]
fn transpose([a, b, c, d]: [u32x4; 4]) -> [u32x4; 4] {
    let wide64 = |x: u32x4| -> u64x2 { bytemuck::cast(x) };
    let narrow = |x: u64x2| -> u32x4 { bytemuck::cast(x) };
    let (ab_lo, ab_hi) = (wide64(a.unpack_lo(b)), wide64(a.unpack_hi(b)));
    let (cd_lo, cd_hi) = (wide64(c.unpack_lo(d)), wide64(c.unpack_hi(d)));
    [
        narrow(ab_lo.unpack_lo(cd_lo)),
        narrow(ab_lo.unpack_hi(cd_lo)),
        narrow(ab_hi.unpack_lo(cd_hi)),
        narrow(ab_hi.unpack_hi(cd_hi)),
    ]
}

/// Low and high words of the four 32x32 to 64-bit products.
#[cfg(not(feature = "simd-intrinsics"))]
#[inline(always)]
fn mul_wide(a: u32x4, b: u32x4) -> (u32x4, u32x4) {
    (a * b, a.mul_keep_high(b))
}

#[cfg(feature = "simd-intrinsics")]
use arch::mul_wide;

#[inline(always)]
fn rotl(x: u32x4, r: u32) -> u32x4 {
    x.unbounded_shl_scalar(r) | x.unbounded_shr_scalar(32 - r)
}

impl Quad {
    #[inline(always)]
    fn step(&mut self) {
        let Quad { o, h } = self;
        let m0 = h[0] | ONE4;
        let m1 = h[1] | ONE4;
        let (lo0, hi0) = mul_wide(o[0], m0);
        let (lo1, hi1) = mul_wide(o[2], m1);
        let n0 = o[1] ^ hi1 ^ lo1;
        let n1 = rotl(lo1, 16) ^ h[2];
        let n2 = o[3] ^ hi0 ^ lo0;
        let n3 = rotl(lo0, 16) ^ h[3];
        h[0] ^= rotl(h[1], 7);
        h[1] ^= rotl(h[2], 13);
        h[2] ^= rotl(h[3], 22);
        h[3] ^= rotl(h[0], 3);
        h[0] = (h[0] + WEYL4) ^ n0;
        *o = [n0, n1, n2, n3];
    }

    /// `F` on chunks `c0 .. c0 + 4` at once.
    // The rounds run on a local and swap by value: `mem::swap` of the fields made LLVM split
    // every vector into 64-bit halves. Out of line, the row loop kept state on the stack.
    #[inline(always)]
    fn seed(&mut self, key: &[u32; 4], c0: u64) {
        let lo = c0 as u32;
        let mut q = Quad {
            o: [
                u32x4::new([lo, lo + 1, lo + 2, lo + 3]),
                u32x4::splat((c0 >> 32) as u32),
                DOMAIN_STREAM4,
                AUX_STREAM4,
            ],
            h: key.map(u32x4::splat),
        };
        for rc in &RC4 {
            q.step();
            q.o[0] ^= *rc;
            q = Quad { o: q.h, h: q.o };
        }
        *self = q;
    }
}

impl Lanes {
    fn load(o: &[[u32; 8]; 4], h: &[[u32; 8]; 4]) -> Lanes {
        let quad = |k: usize| Quad {
            o: o.map(|w| u32x4::new([w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3]])),
            h: h.map(|w| u32x4::new([w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3]])),
        };
        Lanes {
            q: [quad(0), quad(1)],
        }
    }

    fn save(&self, o: &mut [[u32; 8]; 4], h: &mut [[u32; 8]; 4]) {
        for (k, q) in self.q.iter().enumerate() {
            for w in 0..4 {
                o[w][4 * k..4 * k + 4].copy_from_slice(&q.o[w].to_array());
                h[w][4 * k..4 * k + 4].copy_from_slice(&q.h[w].to_array());
            }
        }
    }

    #[inline(always)]
    fn step(&mut self) {
        self.q[0].step();
        self.q[1].step();
    }

    /// `F` on the eight chunks of group `g` at once.
    #[inline(always)]
    fn seed(&mut self, key: &[u32; 4], g: u64) {
        self.q[0].seed(key, 8 * g);
        self.q[1].seed(key, 8 * g + 4);
    }

    /// The row's eight blocks in stream order: each quad's blocks are the transpose of its
    /// four word vectors, done as 32-bit then 64-bit interleaves.
    #[inline(always)]
    fn blocks(&self) -> [u32x4; 8] {
        let [a, b] = self.q.map(|q| transpose(q.o));
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }
}

// ---- Reads and conversions ----------------------------------------------------------------

#[inline(always)]
pub(crate) fn align(pos: u64, w: u32) -> u64 {
    let w = u64::from(w);
    (pos + w - 1) & !(w - 1)
}

#[inline(always)]
pub(crate) fn to_f64(raw: u64) -> f64 {
    (raw >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[inline(always)]
pub(crate) fn to_f32(raw: u32) -> f32 {
    (raw >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

/// `(raw >> 5) * 2^-11` as binary16 bits. The significand is the 11-bit integer `k`, so the
/// encoding is exact: zero, or a normal half with exponent `floor(log2 k) + 4`.
fn to_f16_bits(raw: u16) -> u16 {
    let k = u32::from(raw >> 5);
    if k == 0 {
        return 0;
    }
    let m = 31 - k.leading_zeros();
    (((m + 4) << 10) | ((k << (10 - m)) & 0x3ff)) as u16
}

/// `floor(raw * 1112064 / 2^64)`, then skip the surrogate range.
fn to_char(raw: u64) -> char {
    let u = ((u128::from(raw) * 1_112_064) >> 64) as u32;
    let u = if u < 0xd800 { u } else { u + 0x800 };
    char::from_u32(u).expect("below 0x110000 and not a surrogate")
}

/// A type a fill can produce from `BITS` aligned stream bits.
///
/// After alignment every fill is the same little-endian byte stream, so a row is stored as
/// its 128 bytes and then fixed up in place: a byte swap on big-endian hosts for integers,
/// the float mapping for floats.
trait Elem: bytemuck::Pod {
    const BITS: u32;
    fn from_raw(lo: u64, hi: u64) -> Self;
    fn fix(&mut self);

    /// Write one row of `1024 / BITS` elements, given as the row's eight blocks.
    #[inline(always)]
    fn store_row(blocks: &[u32x4; 8], dst: &mut [Self]) {
        store_le(blocks, dst)
    }
}

/// The default row store: the bytes, then the fix-up.
#[inline(always)]
pub(crate) fn store_le<T: Elem>(blocks: &[u32x4; 8], dst: &mut [T]) {
    // A fixed length lets the stores go straight to `dst`, not through a stack copy.
    let bytes: &mut [u8; 128] = bytemuck::cast_slice_mut(dst)
        .try_into()
        .expect("a row is 128 bytes");
    for (block, out) in blocks.iter().zip(bytes.as_chunks_mut::<16>().0) {
        out.copy_from_slice(bytemuck::bytes_of(&block.to_array().map(u32::to_le)));
    }
    for x in dst {
        x.fix();
    }
}

macro_rules! int_elem {
    ($($ty:ty, $bits:expr;)*) => {$(
        impl Elem for $ty {
            const BITS: u32 = $bits;
            #[inline(always)]
            fn from_raw(lo: u64, hi: u64) -> Self {
                let _ = hi;
                lo as $ty
            }
            #[inline(always)]
            fn fix(&mut self) {
                *self = Self::from_le(*self);
            }
        }
    )*};
}

int_elem! {
    u8, 8;
    u16, 16;
    u32, 32;
    u64, 64;
}

impl Elem for u128 {
    const BITS: u32 = 128;
    #[inline(always)]
    fn from_raw(lo: u64, hi: u64) -> Self {
        u128::from(lo) | u128::from(hi) << 64
    }
    #[inline(always)]
    fn fix(&mut self) {
        *self = Self::from_le(*self);
    }
}

impl Elem for f32 {
    const BITS: u32 = 32;
    #[inline(always)]
    fn from_raw(lo: u64, _hi: u64) -> Self {
        to_f32(lo as u32)
    }
    #[inline(always)]
    fn fix(&mut self) {
        *self = to_f32(u32::from_le(self.to_bits()));
    }
    #[cfg(feature = "simd-intrinsics")]
    #[inline(always)]
    fn store_row(blocks: &[u32x4; 8], dst: &mut [f32]) {
        arch::store_f32(blocks, dst.try_into().expect("a row is 32 floats"))
    }
}

impl Elem for f64 {
    const BITS: u32 = 64;
    #[inline(always)]
    fn from_raw(lo: u64, _hi: u64) -> Self {
        to_f64(lo)
    }
    #[inline(always)]
    fn fix(&mut self) {
        *self = to_f64(u64::from_le(self.to_bits()));
    }
    #[cfg(feature = "simd-intrinsics")]
    #[inline(always)]
    fn store_row(blocks: &[u32x4; 8], dst: &mut [f64]) {
        arch::store_f64(blocks, dst.try_into().expect("a row is 16 doubles"))
    }
}

// ---- The generator ------------------------------------------------------------------------

/// A Tandem8x32 generator: key, bit position, chunk length, and a cache of the current row.
///
/// Equality and `Debug` cover the transport form only. Every method that draws aligns the
/// position to the width of the value, reads, and advances past it, as the specification
/// requires, so draws of mixed widths from one generator agree with the other references.
// The C layout keeps `k` and `cached` between `pos` and `row`. Otherwise the compiler pairs
// the loads of `pos` and `row` into one 16-byte load right after the 8-byte store of `pos`,
// which defeats store forwarding and doubles the cost of a scalar draw.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Tandem {
    key: [u32; 4],
    pos: u64,
    k: u32,
    cached: bool,
    row: u64,
    o: [[u32; 8]; 4],
    h: [[u32; 8]; 4],
}

impl Tandem {
    /// A generator from a 128-bit integer seed through the specification's seed whitening,
    /// at position 0 with the default chunk length.
    pub fn new(seed: u128) -> Self {
        Self::with_chunk_length(seed, DEFAULT_K)
    }

    /// Like [`new`](Self::new), with chunk length `k`.
    ///
    /// # Panics
    ///
    /// If `k` is not a power of two in `1..=65536`.
    pub fn with_chunk_length(seed: u128, k: u32) -> Self {
        let mut o = [0, 0, DOMAIN_SEED, 0];
        let mut h = [
            seed as u32,
            (seed >> 32) as u32,
            (seed >> 64) as u32,
            (seed >> 96) as u32,
        ];
        f(&mut o, &mut h);
        Self::from_key(o, 0, k)
    }

    /// A generator from its transport form: raw key, bit position, and chunk length.
    ///
    /// # Panics
    ///
    /// If `k` is not a power of two in `1..=65536`.
    pub fn from_key(key: [u32; 4], position: u64, k: u32) -> Self {
        assert!(
            k.is_power_of_two() && k <= 65536,
            "chunk length must be a power of two in 1..=65536"
        );
        Tandem {
            key,
            pos: position,
            k,
            cached: false,
            row: 0,
            o: [[0; 8]; 4],
            h: [[0; 8]; 4],
        }
    }

    /// The 128-bit key as four words.
    pub fn key(&self) -> [u32; 4] {
        self.key
    }

    /// The stream bit position of the next draw, before alignment.
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// Move to a stream bit position.
    pub fn set_position(&mut self, position: u64) {
        self.pos = position;
    }

    /// The chunk length `K`.
    pub fn chunk_length(&self) -> u32 {
        self.k
    }

    // Rows -------------------------------------------------------------------------------

    /// Produce rows `row .. row + nrows` in stream order through `sink`, or only move the
    /// cache when there is none. Stepping forward
    /// inside the cached group costs one step per row; any other jump reseeds the group.
    /// Afterwards the cache holds the last row produced.
    #[inline(never)]
    fn run_rows<S: FnMut(&[u32x4; 8])>(
        &mut self,
        mut row: u64,
        mut nrows: u64,
        mut sink: Option<S>,
    ) {
        if nrows == 0 {
            return;
        }
        let k = u64::from(self.k);
        let shift = self.k.trailing_zeros();
        let mask = k - 1;
        let mut lanes = Lanes::load(&self.o, &self.h);
        let mut at = self.row;
        let mut live = self.cached;
        while nrows > 0 {
            if !live || at != row {
                if live && row > at && row >> shift == at >> shift {
                    for _ in at..row {
                        lanes.step();
                    }
                } else {
                    lanes.seed(&self.key, row >> shift);
                    for _ in 0..=(row & mask) {
                        lanes.step();
                    }
                }
                live = true;
            }
            let run = (k - (row & mask)).min(nrows);
            for r in 0..run {
                if r > 0 {
                    lanes.step();
                }
                if let Some(sink) = &mut sink {
                    sink(&lanes.blocks());
                }
            }
            at = row + run - 1;
            row += run;
            nrows -= run;
        }
        lanes.save(&mut self.o, &mut self.h);
        self.row = at;
        self.cached = true;
    }

    #[inline(always)]
    fn load_row(&mut self, row: u64) {
        if !self.cached || self.row != row {
            self.run_rows(row, 1, None::<fn(&[u32x4; 8])>);
        }
    }

    #[inline(always)]
    fn word_at(&self, p: u64) -> u32 {
        self.o[((p >> 5) & 3) as usize][((p >> 7) & 7) as usize]
    }

    /// `w` bits (a power of two, 1 to 64) at the aligned position `p`.
    #[inline(always)]
    fn read(&mut self, p: u64, w: u32) -> u64 {
        self.load_row(p >> 10);
        if w == 64 {
            u64::from(self.word_at(p)) | u64::from(self.word_at(p + 32)) << 32
        } else {
            u64::from((self.word_at(p) >> (p & 31)) & (u32::MAX >> (32 - w)))
        }
    }

    /// `w` bits (a power of two, 1 to 128) at the aligned position `p`, as (low, high) words.
    #[inline(always)]
    fn read_wide(&mut self, p: u64, w: u32) -> (u64, u64) {
        if w == 128 {
            (self.read(p, 64), self.read(p + 64, 64))
        } else {
            (self.read(p, w), 0)
        }
    }

    /// Draw `w` bits: align, advance, read.
    #[inline(always)]
    fn next(&mut self, w: u32) -> u64 {
        let p = align(self.pos, w);
        self.pos = p + u64::from(w);
        self.read(p, w)
    }

    // Scalar draws -----------------------------------------------------------------------

    /// One stream bit.
    pub fn next_bool(&mut self) -> bool {
        self.next(1) != 0
    }
    /// An aligned 8-bit draw.
    pub fn next_u8(&mut self) -> u8 {
        self.next(8) as u8
    }
    /// An aligned 16-bit draw.
    pub fn next_u16(&mut self) -> u16 {
        self.next(16) as u16
    }
    /// An aligned 32-bit draw.
    pub fn next_u32(&mut self) -> u32 {
        self.next(32) as u32
    }
    /// An aligned 64-bit draw.
    pub fn next_u64(&mut self) -> u64 {
        self.next(64)
    }
    /// An aligned 128-bit draw.
    pub fn next_u128(&mut self) -> u128 {
        let p = align(self.pos, 128);
        self.pos = p + 128;
        let (lo, hi) = self.read_wide(p, 128);
        u128::from(lo) | u128::from(hi) << 64
    }
    /// A uniform `f32` in `[0, 1)` with 24 random bits: `(raw >> 8) * 2^-24`.
    pub fn next_f32(&mut self) -> f32 {
        to_f32(self.next_u32())
    }
    /// A uniform `f64` in `[0, 1)` with 53 random bits: `(raw >> 11) * 2^-53`.
    pub fn next_f64(&mut self) -> f64 {
        to_f64(self.next_u64())
    }
    /// The IEEE binary16 bit pattern of a uniform draw in `[0, 1)`: `(raw >> 5) * 2^-11`.
    pub fn next_f16_bits(&mut self) -> u16 {
        to_f16_bits(self.next_u16())
    }
    /// A complex `f32` as `[re, im]`: two `f32` draws, real part first.
    pub fn next_c32(&mut self) -> [f32; 2] {
        [self.next_f32(), self.next_f32()]
    }
    /// A complex `f64` as `[re, im]`: two `f64` draws, real part first.
    pub fn next_c64(&mut self) -> [f64; 2] {
        [self.next_f64(), self.next_f64()]
    }
    /// A uniform Unicode scalar value, from 64 stream bits.
    pub fn next_char(&mut self) -> char {
        to_char(self.next_u64())
    }

    // Fills ------------------------------------------------------------------------------

    /// Fill `out` with the values that `out.len()` scalar draws would produce. After
    /// alignment the stream is read in whole rows, so this is the fast path.
    fn fill<T: Elem>(&mut self, out: &mut [T]) {
        let w = T::BITS;
        let per_row = (1024 / w) as usize;
        let mut p = align(self.pos, w);
        self.pos = p + u64::from(w) * out.len() as u64;

        let to_row = ((1024 - (p & 1023)) / u64::from(w)) as usize % per_row;
        let head = to_row.min(out.len());
        let (head_out, rest) = out.split_at_mut(head);
        for x in head_out {
            let (lo, hi) = self.read_wide(p, w);
            *x = T::from_raw(lo, hi);
            p += u64::from(w);
        }

        let nrows = rest.len() / per_row;
        let (rows, tail) = rest.split_at_mut(nrows * per_row);
        let mut chunks = rows.chunks_exact_mut(per_row);
        self.run_rows(
            p >> 10,
            nrows as u64,
            Some(|blocks: &[u32x4; 8]| {
                T::store_row(blocks, chunks.next().expect("one chunk per row"));
            }),
        );
        p += 1024 * nrows as u64;

        for x in tail {
            let (lo, hi) = self.read_wide(p, w);
            *x = T::from_raw(lo, hi);
            p += u64::from(w);
        }
    }

    /// Fill with 8-bit draws.
    pub fn fill_u8(&mut self, out: &mut [u8]) {
        self.fill(out)
    }
    /// Fill with 16-bit draws.
    pub fn fill_u16(&mut self, out: &mut [u16]) {
        self.fill(out)
    }
    /// Fill with 32-bit draws.
    pub fn fill_u32(&mut self, out: &mut [u32]) {
        self.fill(out)
    }
    /// Fill with 64-bit draws.
    pub fn fill_u64(&mut self, out: &mut [u64]) {
        self.fill(out)
    }
    /// Fill with 128-bit draws.
    pub fn fill_u128(&mut self, out: &mut [u128]) {
        self.fill(out)
    }
    /// Fill with uniform `f32` in `[0, 1)`.
    pub fn fill_f32(&mut self, out: &mut [f32]) {
        self.fill(out)
    }
    /// Fill with uniform `f64` in `[0, 1)`.
    pub fn fill_f64(&mut self, out: &mut [f64]) {
        self.fill(out)
    }
    /// Fill with complex `f32` as `[re, im]`: the `f32` fill of length `2 * out.len()`.
    pub fn fill_c32(&mut self, out: &mut [[f32; 2]]) {
        self.fill_f32(out.as_flattened_mut())
    }
    /// Fill with complex `f64` as `[re, im]`: the `f64` fill of length `2 * out.len()`.
    pub fn fill_c64(&mut self, out: &mut [[f64; 2]]) {
        self.fill_f64(out.as_flattened_mut())
    }
    /// Fill with binary16 bit patterns of uniform draws in `[0, 1)`.
    pub fn fill_f16_bits(&mut self, out: &mut [u16]) {
        self.fill(out);
        for x in out {
            *x = to_f16_bits(*x);
        }
    }
    /// Fill with single stream bits.
    pub fn fill_bool(&mut self, out: &mut [bool]) {
        for x in out {
            *x = self.next_bool();
        }
    }
    /// Fill with uniform Unicode scalar values.
    pub fn fill_char(&mut self, out: &mut [char]) {
        for x in out {
            *x = self.next_char();
        }
    }

    // Random access ----------------------------------------------------------------------

    /// Element `i` of the fill that would start here, without advancing. Works on a copy,
    /// so the cache stays where it is.
    fn at(&self, i: u64, w: u32) -> u64 {
        let mut tmp = *self;
        tmp.read(align(self.pos, w) + i * u64::from(w), w)
    }
    /// Element `i` of the `u32` fill that would start here, without advancing.
    pub fn at_u32(&self, i: u64) -> u32 {
        self.at(i, 32) as u32
    }
    /// Element `i` of the `u64` fill that would start here, without advancing.
    pub fn at_u64(&self, i: u64) -> u64 {
        self.at(i, 64)
    }
    /// Element `i` of the `f32` fill that would start here, without advancing.
    pub fn at_f32(&self, i: u64) -> f32 {
        to_f32(self.at_u32(i))
    }
    /// Element `i` of the `f64` fill that would start here, without advancing.
    pub fn at_f64(&self, i: u64) -> f64 {
        to_f64(self.at_u64(i))
    }

    // Derived generators -----------------------------------------------------------------

    fn child(&self, counter: u64, domain: u32, aux: u32, hidden: bool) -> Tandem {
        let (o, h) = f_keyed(&self.key, counter, domain, aux);
        Tandem::from_key(if hidden { h } else { o }, 0, self.k)
    }

    /// Child `index` by key alone: the same child for the same index, whatever the position.
    pub fn split(&self, index: u64) -> Tandem {
        self.child(index >> 1, DOMAIN_SPLIT, 0, index & 1 == 1)
    }

    /// A generator for a named purpose, by key alone.
    pub fn sub(&self, purpose: u64) -> Tandem {
        self.child(purpose, DOMAIN_FOLD, 0, false)
    }

    /// Fork `n` children from the current block and move past it. The children depend on
    /// the block the parent is in, so successive forks give fresh children. The parent
    /// advances once per call, for an empty batch too.
    pub fn fork(&mut self, n: u64) -> Fork {
        let b = self.pos >> 7;
        self.pos = (b + 1) << 7;
        Fork {
            parent: *self,
            block: b,
            next: 0,
            end: n,
        }
    }
}

impl PartialEq for Tandem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.pos == other.pos && self.k == other.k
    }
}

impl Eq for Tandem {}

impl core::fmt::Debug for Tandem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tandem")
            .field("key", &self.key)
            .field("position", &self.pos)
            .field("chunk_length", &self.k)
            .finish()
    }
}

/// The children of a [`Tandem::fork`], in index order.
#[derive(Clone, Debug)]
pub struct Fork {
    parent: Tandem,
    block: u64,
    next: u64,
    end: u64,
}

impl Iterator for Fork {
    type Item = Tandem;

    fn next(&mut self) -> Option<Tandem> {
        if self.next == self.end {
            return None;
        }
        let i = self.next;
        self.next += 1;
        Some(
            self.parent
                .child(self.block, DOMAIN_FORK, (i >> 1) as u32, i & 1 == 1),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = (self.end - self.next) as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Fork {}
