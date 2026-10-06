//! Readers for the spec's conformance files in tests/conformance, byte-identical copies of
//! tandem-spec f420545 conformance/*.json. CI checks the copies.
#![allow(dead_code)]

use serde_json::Value;
use tandem_rng::Tandem;

pub fn cases(file: &str) -> Vec<Value> {
    load(file)["cases"].as_array().expect("cases").clone()
}

pub fn load(file: &str) -> Value {
    let path = format!("{}/tests/conformance/{file}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).expect(file)
}

/// The case whose `id` ends with `name`, such as `CROSS_BELOW32[4]`.
pub fn case<'a>(cases: &'a [Value], name: &str) -> &'a Value {
    cases
        .iter()
        .find(|c| c["id"].as_str().unwrap().ends_with(name))
        .unwrap_or_else(|| panic!("no case {name}"))
}

pub fn hex(v: &Value) -> u64 {
    u64::from_str_radix(v.as_str().expect("hex string"), 16).expect("hex")
}

pub fn hexes(v: &Value) -> Vec<u64> {
    v.as_array().expect("array").iter().map(hex).collect()
}

pub fn int(c: &Value, field: &str) -> u64 {
    c[field].as_u64().unwrap_or_else(|| panic!("{field}"))
}

pub fn n(c: &Value) -> usize {
    int(c, "n") as usize
}

pub fn id(c: &Value) -> &str {
    c["id"].as_str().unwrap()
}

pub fn key(c: &Value) -> [u32; 4] {
    let k = hexes(&c["key"]);
    [k[0], k[1], k[2], k[3]].map(|w| w as u32)
}

/// The generator of a case at its start.
pub fn rng(c: &Value) -> Tandem {
    Tandem::from_key(key(c), int(c, "start"), int(c, "K") as u32)
}

pub fn f64s(c: &Value) -> Vec<f64> {
    hexes(&c["values"])
        .into_iter()
        .map(f64::from_bits)
        .collect()
}

pub fn f32s(c: &Value) -> Vec<f32> {
    hexes(&c["values"])
        .into_iter()
        .map(|b| f32::from_bits(b as u32))
        .collect()
}

/// Bit equality with `std`, whose fused multiply-add is tandem-c's. Without it the plain
/// `a * b + c` differs in the last bits.
pub fn same_f64(got: f64, want: f64) -> bool {
    if cfg!(feature = "std") {
        got.to_bits() == want.to_bits()
    } else {
        (got - want).abs() <= 1e-12 * want.abs() + 1e-15
    }
}

/// Bit equality with `std`, else the tolerance `tol` of the conformance files.
pub fn same_f32(got: f32, want: f32) -> bool {
    if cfg!(feature = "std") {
        got.to_bits() == want.to_bits()
    } else {
        (got - want).abs() <= 16.0 * f32::EPSILON * want.abs() + 1e-6
    }
}
