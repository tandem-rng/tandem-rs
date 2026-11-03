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
        // Transport positions are unsigned successors, not constructor starts below 2^63.
        if !(t.chunk_length.is_power_of_two() && t.chunk_length <= 65536) {
            return Err(serde::de::Error::custom(Invalid::Length(t.chunk_length)));
        }
        Ok(Tandem::unchecked(t.key, t.position, t.chunk_length))
    }
}

enum Invalid {
    Length(u32),
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Invalid::Length(k) => write!(f, "chunk length {k} is not a power of two in 1..=65536"),
        }
    }
}
