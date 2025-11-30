// This module implements a binary arithmetic (de)coder (bac) as specified in
// Section 13 of the VP8 specification.

// ----------------------------------------------------------------------------
#[derive(Clone, Copy)]
struct State {
    value: u64,
    range: u32,
    bits_valid: i32,
}

// ----------------------------------------------------------------------------
pub struct ArithmeticDecoder<'a> {
    chunks: &'a [[u8; 4]],
    last_chunk: [u8; 4],
    chunk_index: usize,
    chunk_count: usize,
    state: State,
}

impl<'a> ArithmeticDecoder<'a> {
    // ------------------------------------------------------------------------
    pub fn new(data: &'a [u8]) -> Self {
        let (chunks, last) = data.as_chunks::<4>();
        let mut last_chunk = [0u8; 4];
        last_chunk[..last.len()].copy_from_slice(last);
        let chunk_count = data.len().div_ceil(4);

        let state = State {
            value: 0,
            range: 255,
            bits_valid: -8,
        };

        Self {
            chunks,
            last_chunk,
            chunk_index: 0,
            chunk_count,
            state,
        }
    }

    // ------------------------------------------------------------------------
    pub fn is_overflow(&self) -> bool {
        self.chunk_index > self.chunk_count
    }

    // ------------------------------------------------------------------------
    fn refill_bits(&mut self) {
        let chunk = self
            .chunks
            .get(self.chunk_index)
            .copied()
            .unwrap_or(self.last_chunk);
        self.chunk_index += 1;
        self.state.value = (self.state.value << 32) | u64::from(u32::from_be_bytes(chunk));
        self.state.bits_valid += 32;
    }

    // ------------------------------------------------------------------------
    // uncomment the below line to disable inlining for easier profiling
    //#[inline(never)]
    pub fn read_bit(&mut self, probability: u32) -> bool {
        if self.state.bits_valid < 0 {
            self.refill_bits();
        }

        let split = 1 + (((self.state.range - 1) * probability) >> 8);
        let value_split = u64::from(split) << self.state.bits_valid;
        let value = self.state.value.checked_sub(value_split);

        if let Some(value) = value {
            self.state.range -= split;
            self.state.value = value;
        } else {
            self.state.range = split;
        }

        let range_8 = self.state.range as u8;
        let shift = range_8.leading_zeros();
        self.state.range <<= shift;
        self.state.bits_valid -= shift as i32;

        value.is_some()
    }

    // ------------------------------------------------------------------------
    // uncomment the below line to disable inlining for easier profiling
    //#[inline(never)]
    pub fn read_flag(&mut self) -> bool {
        if self.state.bits_valid < 0 {
            self.refill_bits();
        }

        let split = 1 + ((self.state.range - 1) >> 1);
        let value_split = u64::from(split) << self.state.bits_valid;
        let value = self.state.value.checked_sub(value_split);

        if let Some(value) = value {
            self.state.range -= split;
            self.state.value = value;
        } else {
            self.state.range = split;
        }

        let shift = if self.state.range == 0x80 { 0 } else { 1 };
        self.state.range <<= shift;
        self.state.bits_valid -= shift;

        value.is_some()
    }

    // ------------------------------------------------------------------------
    // uncomment the below line to disable inlining for easier profiling
    //#[inline(never)]
    pub fn read_signed(&mut self, abs_value: i32) -> i32 {
        if self.state.bits_valid < 0 {
            self.refill_bits();
        }

        let split_32 = (self.state.range + 1) >> 1;
        let split_64 = u64::from(split_32) << self.state.bits_valid;
        let value = self.state.value.checked_sub(split_64);

        if let Some(value) = value {
            self.state.range -= split_32;
            self.state.value = value;
        } else {
            self.state.range = split_32;
        }

        self.state.range <<= 1;
        self.state.bits_valid -= 1;

        if value.is_some() {
            -abs_value
        } else {
            abs_value
        }
    }

    // ------------------------------------------------------------------------
    pub fn read_bool(&mut self, probability: u8) -> bool {
        self.read_bit(probability as u32)
    }

    // ------------------------------------------------------------------------
    pub fn read_literal(&mut self, n: u8) -> u8 {
        (0..n).fold(0u8, |v, _| (v << 1) | u8::from(self.read_flag()))
    }

    // ------------------------------------------------------------------------
    pub fn read_signed_value(&mut self, n: u8) -> i32 {
        let flag = self.read_flag();
        if !flag {
            return 0;
        }
        let magnitude = self.read_literal(n) as i32;
        self.read_signed(magnitude)
    }
}

// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_decoder_hello_short() {
        let data = b"hel";
        let mut decoder = ArithmeticDecoder::new(data);
        assert!(!decoder.read_flag());
        assert!(decoder.read_bool(10));
        assert!(!decoder.read_bool(250));
        assert_eq!(1, decoder.read_literal(1));
        assert_eq!(5, decoder.read_literal(3));
        assert_eq!(64, decoder.read_literal(8));
        assert_eq!(185, decoder.read_literal(8));
        assert!(!decoder.is_overflow());
    }

    #[test]
    fn test_arithmetic_decoder_hello_long() {
        let data = b"hello world";
        let mut decoder = ArithmeticDecoder::new(data);
        assert!(!decoder.read_flag());
        assert!(decoder.read_bool(10));
        assert!(!decoder.read_bool(250));
        assert_eq!(1, decoder.read_literal(1));
        assert_eq!(5, decoder.read_literal(3));
        assert_eq!(64, decoder.read_literal(8));
        assert_eq!(185, decoder.read_literal(8));
        assert_eq!(31, decoder.read_literal(8));
        assert!(!decoder.is_overflow());
    }

    #[test]
    fn test_arithmetic_decoder_uninit() {
        let data = b"";
        let mut decoder = ArithmeticDecoder::new(data);
        let _ = decoder.read_flag();
        assert!(decoder.is_overflow());
    }
}
