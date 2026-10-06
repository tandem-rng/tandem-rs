//! The cuRAND baseline of the GPU table: Philox4x32-10 words into device memory, timed as
//! `bench_gpu` times its fills. cuRAND and the CUDA runtime load at run time, so the crate keeps
//! no CUDA dependency. Run with both libraries on the loader path:
//! cargo run --release --example bench_curand [log2 words, default 26]

use std::ffi::{CStr, c_char, c_int, c_void};
use std::time::Instant;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;
const CURAND_RNG_PSEUDO_PHILOX4_32_10: c_int = 161;

/// The first of `names` that loads.
fn open(names: &[&CStr]) -> Option<*mut c_void> {
    names
        .iter()
        .map(|name| unsafe { dlopen(name.as_ptr(), RTLD_NOW) })
        .find(|h| !h.is_null())
}

/// Symbol `name` of `lib` as a function pointer of type `F`.
///
/// # Safety
/// `F` must be the C signature of the symbol.
unsafe fn sym<F: Copy>(lib: *mut c_void, name: &CStr) -> F {
    let p = unsafe { dlsym(lib, name.as_ptr()) };
    assert!(!p.is_null(), "missing symbol {name:?}");
    unsafe { std::mem::transmute_copy::<*mut c_void, F>(&p) }
}

fn main() {
    let (Some(curand), Some(cudart)) = (
        open(&[c"libcurand.so.10", c"libcurand.so"]),
        open(&[c"libcudart.so", c"libcudart.so.13", c"libcudart.so.12"]),
    ) else {
        eprintln!("no libcurand or libcudart on the loader path");
        return;
    };
    let log2n: u32 = std::env::args()
        .nth(1)
        .map_or(26, |a| a.parse().expect("log2 of the word count"));
    let n = 1usize << log2n;

    type Malloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
    type Sync = unsafe extern "C" fn() -> c_int;
    type Create = unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int;
    type Seed = unsafe extern "C" fn(*mut c_void, u64) -> c_int;
    type Generate = unsafe extern "C" fn(*mut c_void, *mut u32, usize) -> c_int;
    let (malloc, sync, create, seed, generate) = unsafe {
        (
            sym::<Malloc>(cudart, c"cudaMalloc"),
            sym::<Sync>(cudart, c"cudaDeviceSynchronize"),
            sym::<Create>(curand, c"curandCreateGenerator"),
            sym::<Seed>(curand, c"curandSetPseudoRandomGeneratorSeed"),
            sym::<Generate>(curand, c"curandGenerate"),
        )
    };
    let mut out = std::ptr::null_mut();
    let mut generator = std::ptr::null_mut();
    unsafe {
        assert_eq!(malloc(&mut out, n * 4), 0, "cudaMalloc");
        assert_eq!(create(&mut generator, CURAND_RNG_PSEUDO_PHILOX4_32_10), 0);
        assert_eq!(seed(generator, 42), 0);
    }
    let fill = || assert_eq!(unsafe { generate(generator, out.cast(), n) }, 0);
    let wait = || assert_eq!(unsafe { sync() }, 0);

    // The method of bench_gpu: 2 s of warm-up, then the median and the fastest of 21 runs.
    let measure = |fills: usize| {
        let run = || {
            let t0 = Instant::now();
            for _ in 0..fills {
                fill();
            }
            wait();
            t0.elapsed().as_secs_f64() / fills as f64
        };
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < 2.0 {
            run();
        }
        let mut t: Vec<f64> = (0..21).map(|_| run()).collect();
        t.sort_by(f64::total_cmp);
        (t[10], t[0])
    };
    let gibs = |seconds: f64| (n * 4) as f64 / seconds / (1u64 << 30) as f64;
    for (label, fills) in [("one fill per call", 1), ("32 fills back to back", 32)] {
        let (median, fastest) = measure(fills);
        println!(
            "{label:<24} 2^{log2n} words, median (fastest) GiB/s: cuRAND Philox4x32-10 {:7.1} ({:7.1})",
            gibs(median),
            gibs(fastest)
        );
    }
}
