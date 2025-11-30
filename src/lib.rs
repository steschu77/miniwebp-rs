//! Decoding of WebP Images
pub use self::webp_decoder::{Error, read_image};

pub mod vp8;
pub mod vp8_bac;
pub mod vp8_ipred;
pub mod vp8_loop_filter;
pub mod vp8_tokens;
pub mod vp8_xform;
pub mod webp_decoder;
