// WEBP decompression API.
use super::vp8::Vp8Decoder;
use super::vp8::Frame;
use std::io::{self};
use std::ops::Range;

// ----------------------------------------------------------------------------
#[derive(Debug)]
pub enum Error {
    IoError(io::Error),
    InvalidSignature,
    ChunkHeaderInvalid([u8; 4]),
    ChunkMissing,
    ReservedBitSet,
    InvalidAlphaPreprocessing,
    InvalidCompressionMethod,
    AlphaChunkSizeMismatch,
    ImageTooLarge,
    FrameOutsideImage,
    LosslessUnsupported,
    ExtendedUnsupported,
    VersionNumberInvalid(u8),
    InvalidColorCacheBits(u8),
    HuffmanError,
    BitStreamError,
    TransformError,
    BufferUnderrun,
    Vp8MagicInvalid([u8; 3]),
    InvalidImageSize,
    NotEnoughInitData,
    ColorSpaceInvalid(u8),
    LumaPredictionModeInvalid,
    IntraPredictionModeInvalid,
    ChromaPredictionModeInvalid,
    NonKeyframe,
    InvalidParameter,
    MemoryLimitExceeded,
    InvalidChunkSize,
    NoMoreFrames,
}

// ----------------------------------------------------------------------------
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let err = format!("{:?}", self);
        f.write_str(&err)
    }
}

// ----------------------------------------------------------------------------
impl std::error::Error for Error {}

// ----------------------------------------------------------------------------
impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::IoError(err)
    }
}

// ----------------------------------------------------------------------------
impl From<std::array::TryFromSliceError> for Error {
    fn from(_: std::array::TryFromSliceError) -> Self {
        Error::BufferUnderrun
    }
}

// ----------------------------------------------------------------------------
pub type Result<T> = std::result::Result<T, Error>;

// ----------------------------------------------------------------------------
pub fn read_image(data: &[u8]) -> Result<Frame> {
    let range = read_chunks(data)?;
    let decoder = Vp8Decoder::new();
    let data = data[range].to_vec();
    let frame = decoder.decode_frame(&data)?;
    Ok(frame)
}

// ----------------------------------------------------------------------------
fn read_chunks(data: &[u8]) -> Result<Range<usize>> {
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WEBP" {
        return Err(Error::InvalidSignature);
    }

    let chunk = &data[12..];
    let chunk_fcc = chunk[0..4].try_into()?;
    let chunk_size = u32::from_le_bytes(chunk[4..8].try_into()?) as usize;
    let range = 20..20 + chunk_size;

    match &chunk_fcc {
        b"VP8 " => Ok(range),
        b"VP8L" => Err(Error::LosslessUnsupported),
        b"VP8X" => Err(Error::ExtendedUnsupported),
        _ => Err(Error::ChunkHeaderInvalid(chunk_fcc)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_with_overflow_size() {
        let bytes = vec![
            0x52, 0x49, 0x46, 0x46, 0xaf, 0x37, 0x80, 0x47, 0x57, 0x45, 0x42, 0x50, 0x6c, 0x64,
            0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xfb, 0x7e, 0x73, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
            0x40, 0xfb, 0xff, 0xff, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65, 0x65,
            0x00, 0x00, 0x00, 0x00, 0x62, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49,
            0x49, 0x54, 0x55, 0x50, 0x4c, 0x54, 0x59, 0x50, 0x45, 0x33, 0x37, 0x44, 0x4d, 0x46,
        ];

        let _ = read_image(&bytes);
    }

    #[test]
    fn decode_2x2_single_color_image() {
        // Image data created from imagemagick and output of xxd:
        // $ convert -size 2x2 xc:#f00 red.webp
        // $ xxd -g 1 red.webp | head

        // 2x2 red pixel image
        let webp_2x2 = vec![
            0x52, 0x49, 0x46, 0x46, 0x3c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50,
            0x38, 0x20, 0x30, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x02, 0x00,
            0x02, 0x00, 0x02, 0x00, 0x34, 0x25, 0xa0, 0x02, 0x74, 0xba, 0x01, 0xf8, 0x00, 0x03,
            0xb0, 0x00, 0xfe, 0xf0, 0xc4, 0x0b, 0xff, 0x20, 0xb9, 0x61, 0x75, 0xc8, 0xd7, 0xff,
            0x20, 0x3f, 0xe4, 0x07, 0xfc, 0x80, 0xff, 0xf8, 0xf2, 0x00, 0x00, 0x00,
        ];

        let frame = read_image(&webp_2x2).unwrap();

        // All pixels are the same value
        let first_pixel = &frame.ybuf[0];
        assert!(frame.ybuf.iter().all(|y| y == first_pixel));
    }

    #[test]
    // Test that any odd pixel "tail" is decoded properly
    fn decode_3x3_single_color_image() {

        // 3x3 red pixel image
        let webp_3x3 = vec![
            0x52, 0x49, 0x46, 0x46, 0x3c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50,
            0x38, 0x20, 0x30, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x03, 0x00,
            0x03, 0x00, 0x02, 0x00, 0x34, 0x25, 0xa0, 0x02, 0x74, 0xba, 0x01, 0xf8, 0x00, 0x03,
            0xb0, 0x00, 0xfe, 0xf0, 0xc4, 0x0b, 0xff, 0x20, 0xb9, 0x61, 0x75, 0xc8, 0xd7, 0xff,
            0x20, 0x3f, 0xe4, 0x07, 0xfc, 0x80, 0xff, 0xf8, 0xf2, 0x00, 0x00, 0x00,
        ];

        let frame = read_image(&webp_3x3).unwrap();

        // All pixels are the same value
        let first_pixel = &frame.ybuf[0];
        assert!(frame.ybuf.iter().all(|y| y == first_pixel));
    }
}
