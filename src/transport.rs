//! `serde` support through the transport form: key, bit position, chunk length. The row cache
//! is not part of the stream state, so a deserialized generator continues the same stream.

use core::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Tandem;

#[derive(Serialize, Deserialize)]
#[serde(rename = "Tandem")]
struct Transport {
    key: [u32; 4],
    position: u64,
    chunk_length: u32,
}

impl Serialize for Tandem {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Transport {
            key: self.key(),
            position: self.position(),
            chunk_length: self.chunk_length(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tandem {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let t = Transport::deserialize(deserializer)?;
        // `from_key` panics on a bad length or start, and input data must not be able to panic.
        if !(t.chunk_length.is_power_of_two() && t.chunk_length <= 65536) {
            return Err(serde::de::Error::custom(Invalid::Length(t.chunk_length)));
        }
        if t.position >> 63 != 0 {
            return Err(serde::de::Error::custom(Invalid::Position(t.position)));
        }
        Ok(Tandem::from_key(t.key, t.position, t.chunk_length))
    }
}

enum Invalid {
    Length(u32),
    Position(u64),
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Invalid::Length(k) => write!(f, "chunk length {k} is not a power of two in 1..=65536"),
            Invalid::Position(p) => write!(f, "position {p} is not below 2^63"),
        }
    }
}
