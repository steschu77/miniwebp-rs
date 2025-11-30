// This module implements the subset of VP8 decoding defined in RFC 6386
// that is necessary for decoding WebP images.
//
// # Related Links
// * http://tools.ietf.org/html/rfc6386
// * http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37073.pdf

use super::vp8_bac::ArithmeticDecoder;
use super::{vp8_loop_filter, vp8_xform};
use crate::vp8_ipred::*;
use crate::vp8_tokens::*;
use crate::webp_decoder::{Error, Result};
use core::panic;
use std::default::Default;

// ----------------------------------------------------------------------------
#[derive(Default, Clone, Copy)]
struct Macroblock {
    luma_mode: LumaMode,
    chroma_mode: ChromaMode,
    segmentid: u8,
    coeffs_skipped: bool,
    non_zero_dct: bool,
}

// ----------------------------------------------------------------------------
#[derive(Default, Clone, Copy)]
struct MacroblockContext {
    bpred: [IntraMode; 4],
    complexity: [u8; 9],
}

// ----------------------------------------------------------------------------
/// A Representation of the last decoded video frame
#[derive(Default, Debug, Clone)]
pub struct Frame {
    pub width: usize,
    pub height: usize,

    pub mb_width: usize,
    pub mb_height: usize,

    pub ybuf: Vec<u8>,
    pub ubuf: Vec<u8>,
    pub vbuf: Vec<u8>,

    pub for_display: bool,

    // Section 9.4 and 15
    pub filter_type: bool, //if true uses simple filter // if false uses normal filter
    pub filter_level: u8,
    pub sharpness_level: u8,
    pub first_partition_size: usize,
}

// ----------------------------------------------------------------------------
enum PredictionType {
    I4x4([IntraMode; 16]),
    I16x16(I16x16Mode),
}

// ----------------------------------------------------------------------------
const MAX_SEGMENTS: usize = 4;

#[derive(Clone, Copy, Default)]
struct Segment {
    ydc: i16,
    yac: i16,

    y2dc: i16,
    y2ac: i16,

    uvdc: i16,
    uvac: i16,

    delta_values: bool,

    quantizer_level: i8,
    loopfilter_level: i8,
}

// ----------------------------------------------------------------------------
// 13.3 Different planes to be decoded in DCT coefficient decoding
#[derive(Clone, Copy, PartialEq)]
pub enum Plane {
    /// The Y plane after decoding Y2
    YCoeff1 = 0,
    /// The Y2 plane (specifies the 0th coefficient of the other Y blocks)
    Y2 = 1,
    /// The U or V plane
    Chroma = 2,
    /// The Y plane when there is no Y2 plane
    YCoeff0 = 3,
}

// ----------------------------------------------------------------------------
#[inline]
fn read_le24(b: &[u8; 3]) -> usize {
    usize::from(b[0]) | (usize::from(b[1]) << 8) | (usize::from(b[2]) << 16)
}

// ----------------------------------------------------------------------------
// VP8 Decoder
pub struct Vp8Decoder {
    macroblocks: Vec<Macroblock>,

    frame: Frame,

    segments_enabled: bool,
    segments_update_map: bool,
    segment: [Segment; MAX_SEGMENTS],

    loop_filter_adjustments_enabled: bool,
    ref_delta: [i32; 4],
    mode_delta: [i32; 4],

    num_partitions: usize,

    segment_tree_nodes: [TreeNode; 3],
    token_probs: Box<TokenProbTreeNodes>,

    // Section 9.11
    prob_skip_false: Option<u8>,

    top: Vec<MacroblockContext>,
    left: MacroblockContext,

    // The borders from the previous macroblock, used for predictions
    // See Section 12
    // Note that the left border contains the top left pixel
    top_border_u: Vec<u8>,
    left_border_u: Vec<u8>,

    top_border_v: Vec<u8>,
    left_border_v: Vec<u8>,

    // Debugging / statistics
    pub intra4x4_count: usize,
    pub intra16x16_count: usize,
}

// ----------------------------------------------------------------------------
impl Default for Vp8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
impl Vp8Decoder {
    // ------------------------------------------------------------------------
    pub fn new() -> Self {
        Self {
            macroblocks: Vec::new(),

            frame: Frame::default(),
            segments_enabled: false,
            segments_update_map: false,
            segment: [Segment::default(); MAX_SEGMENTS],

            loop_filter_adjustments_enabled: false,
            ref_delta: [0; 4],
            mode_delta: [0; 4],

            num_partitions: 1,

            segment_tree_nodes: SEGMENT_TREE_NODE_DEFAULTS,
            token_probs: Box::new(COEFF_PROB_NODES),

            // Section 9.11
            prob_skip_false: None,

            top: Vec::new(),
            left: MacroblockContext::default(),

            top_border_u: Vec::new(),
            left_border_u: Vec::new(),

            top_border_v: Vec::new(),
            left_border_v: Vec::new(),

            intra4x4_count: 0,
            intra16x16_count: 0,
        }
    }

    // ------------------------------------------------------------------------
    fn update_token_probabilities<'a>(&mut self, b: &mut ArithmeticDecoder<'a>) -> Result<()> {
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (t, prob) in ks.iter().enumerate().take(NUM_DCT_TOKENS - 1) {
                        if b.read_bool(*prob) {
                            let v = b.read_literal(8);
                            self.token_probs[i][j][k][t].prob = v;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------
    fn read_quantization_indices<'a>(&mut self, b: &mut ArithmeticDecoder<'a>) -> Result<()> {
        fn dc_quant(index: i32) -> i16 {
            #[rustfmt::skip]
            const DC_QUANT: [i16; 128] = [
                4,   5,   6,   7,   8,   9,  10,  10,
                11,  12,  13,  14,  15,  16,  17,  17,
                18,  19,  20,  20,  21,  21,  22,  22,
                23,  23,  24,  25,  25,  26,  27,  28,
                29,  30,  31,  32,  33,  34,  35,  36,
                37,  37,  38,  39,  40,  41,  42,  43,
                44,  45,  46,  46,  47,  48,  49,  50,
                51,  52,  53,  54,  55,  56,  57,  58,
                59,  60,  61,  62,  63,  64,  65,  66,
                67,  68,  69,  70,  71,  72,  73,  74,
                75,  76,  76,  77,  78,  79,  80,  81,
                82,  83,  84,  85,  86,  87,  88,  89,
                91,  93,  95,  96,  98, 100, 101, 102,
                104, 106, 108, 110, 112, 114, 116, 118,
                122, 124, 126, 128, 130, 132, 134, 136,
                138, 140, 143, 145, 148, 151, 154, 157,
            ];

            DC_QUANT[index.clamp(0, 127) as usize]
        }

        fn ac_quant(index: i32) -> i16 {
            #[rustfmt::skip]
            const AC_QUANT: [i16; 128] = [
                4,   5,   6,   7,   8,    9,  10,  11,
                12,  13,  14,  15,  16,  17,  18,  19,
                20,  21,  22,  23,  24,  25,  26,  27,
                28,  29,  30,  31,  32,  33,  34,  35,
                36,  37,  38,  39,  40,  41,  42,  43,
                44,  45,  46,  47,  48,  49,  50,  51,
                52,  53,  54,  55,  56,  57,  58,  60,
                62,  64,  66,  68,  70,  72,  74,  76,
                78,  80,  82,  84,  86,  88,  90,  92,
                94,  96,  98, 100, 102, 104, 106, 108,
                110, 112, 114, 116, 119, 122, 125, 128,
                131, 134, 137, 140, 143, 146, 149, 152,
                155, 158, 161, 164, 167, 170, 173, 177,
                181, 185, 189, 193, 197, 201, 205, 209,
                213, 217, 221, 225, 229, 234, 239, 245,
                249, 254, 259, 264, 269, 274, 279, 284,
            ];
            AC_QUANT[index.clamp(0, 127) as usize]
        }

        let yac_abs = b.read_literal(7);
        let ydc_delta = b.read_signed_value(4);
        let y2dc_delta = b.read_signed_value(4);
        let y2ac_delta = b.read_signed_value(4);
        let uvdc_delta = b.read_signed_value(4);
        let uvac_delta = b.read_signed_value(4);

        let n = if self.segments_enabled {
            MAX_SEGMENTS
        } else {
            1
        };
        for i in 0..n {
            let base = i32::from(if self.segments_enabled {
                if self.segment[i].delta_values {
                    i16::from(self.segment[i].quantizer_level) + i16::from(yac_abs)
                } else {
                    i16::from(self.segment[i].quantizer_level)
                }
            } else {
                i16::from(yac_abs)
            });

            self.segment[i].ydc = dc_quant(base + ydc_delta);
            self.segment[i].yac = ac_quant(base);

            self.segment[i].y2dc = dc_quant(base + y2dc_delta) * 2;
            // The intermediate result (max`284*155`) can be larger than the `i16` range.
            self.segment[i].y2ac = (i32::from(ac_quant(base + y2ac_delta)) * 155 / 100) as i16;

            self.segment[i].uvdc = dc_quant(base + uvdc_delta);
            self.segment[i].uvac = ac_quant(base + uvac_delta);

            if self.segment[i].y2ac < 8 {
                self.segment[i].y2ac = 8;
            }

            if self.segment[i].uvdc > 132 {
                self.segment[i].uvdc = 132;
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------------
    fn read_loop_filter_adjustments<'a>(&mut self, b: &mut ArithmeticDecoder<'a>) -> Result<()> {
        if b.read_flag() {
            for i in 0..4 {
                self.ref_delta[i] = b.read_signed_value(6);
            }

            for i in 0..4 {
                self.mode_delta[i] = b.read_signed_value(6);
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Section 9.3
    fn read_segment_updates<'a>(&mut self, b: &mut ArithmeticDecoder<'a>) -> Result<()> {
        self.segments_update_map = b.read_flag();
        let update_segment_feature_data = b.read_flag();

        if update_segment_feature_data {
            let segment_feature_mode = b.read_flag();

            for i in 0..MAX_SEGMENTS {
                self.segment[i].delta_values = !segment_feature_mode;
            }

            for i in 0..MAX_SEGMENTS {
                self.segment[i].quantizer_level = b.read_signed_value(7) as i8;
            }

            for i in 0..MAX_SEGMENTS {
                self.segment[i].loopfilter_level = b.read_signed_value(6) as i8;
            }
        }

        if self.segments_update_map {
            for i in 0..3 {
                let update = b.read_flag();

                let prob = if update { b.read_literal(8) } else { 255 };
                self.segment_tree_nodes[i].prob = prob;
            }
        }

        Ok(())
    }

    fn read_macroblock_header<'a>(
        &mut self,
        b: &mut ArithmeticDecoder<'a>,
        mbx: usize,
    ) -> Result<(Macroblock, PredictionType)> {
        let mut mb = Macroblock::default();

        if self.segments_enabled && self.segments_update_map {
            mb.segmentid = read_with_tree(b, &self.segment_tree_nodes) as u8;
        };

        mb.coeffs_skipped = if let Some(prob) = self.prob_skip_false {
            b.read_bool(prob)
        } else {
            false
        };

        mb.luma_mode = read_with_typed_tree(b, &TYPED_LUMA_MODE_TREE);

        let pred = match mb.luma_mode.into_intra() {
            // `LumaMode::B` - This is predicted individually
            None => {
                let mut bpred = [IntraMode::DC; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let top = self.top[mbx].bpred[x] as usize;
                        let left = self.left.bpred[y] as usize;
                        let intra = read_with_tree(b, &KEYFRAME_BPRED_MODE_NODES[top][left]) as u8;
                        let bmode =
                            IntraMode::from_i8(intra).ok_or(Error::IntraPredictionModeInvalid)?;
                        bpred[x + y * 4] = bmode;

                        self.top[mbx].bpred[x] = bmode;
                        self.left.bpred[y] = bmode;
                    }
                }
                PredictionType::I4x4(bpred)
            }
            Some(mode) => {
                let i4x4mode = mode.into();
                for i in 0..4 {
                    self.top[mbx].bpred[i] = i4x4mode;
                    self.left.bpred[i] = i4x4mode;
                }
                PredictionType::I16x16(mode)
            }
        };

        mb.chroma_mode = read_with_typed_tree(b, &TYPED_CHROMA_MODE_TREE);

        if b.is_overflow() {
            Err(Error::BufferUnderrun)
        } else {
            Ok((mb, pred))
        }
    }

    fn intra_predict_luma_b_row_0(&mut self, mbx: usize, bpred: [IntraMode; 16], resdata: &[i32]) {
        let ws = create_border_luma_row_0(mbx, self.frame.mb_width, &self.frame.ybuf);
        self.intra_predict_luma_b(ws, mbx, 0, bpred, resdata);
    }

    fn intra_predict_luma_b_row_x(
        &mut self,
        mbx: usize,
        mby: usize,
        bpred: [IntraMode; 16],
        resdata: &[i32],
    ) {
        let ws = create_border_luma_row_x(mbx, mby, self.frame.mb_width, &self.frame.ybuf);
        self.intra_predict_luma_b(ws, mbx, mby, bpred, resdata);
    }

    fn intra_predict_luma_b(
        &mut self,
        mut ws: [u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
        bpred: [IntraMode; 16],
        resdata: &[i32],
    ) {
        self.intra4x4_count += 1;

        let stride = LUMA_STRIDE;
        for (i, (block, mode)) in resdata.chunks(16).take(16).zip(bpred).enumerate() {
            let x0 = 1 + (i % 4) * 4;
            let y0 = 1 + (i / 4) * 4;

            match mode {
                IntraMode::TM => predict_tmpred(&mut ws, 4, x0, y0, stride),
                IntraMode::VE => predict_bvepred(&mut ws, x0, y0, stride),
                IntraMode::HE => predict_bhepred(&mut ws, x0, y0, stride),
                IntraMode::DC => predict_bdcpred(&mut ws, x0, y0, stride),
                IntraMode::LD => predict_bldpred(&mut ws, x0, y0, stride),
                IntraMode::RD => predict_brdpred(&mut ws, x0, y0, stride),
                IntraMode::VR => predict_bvrpred(&mut ws, x0, y0, stride),
                IntraMode::VL => predict_bvlpred(&mut ws, x0, y0, stride),
                IntraMode::HD => predict_bhdpred(&mut ws, x0, y0, stride),
                IntraMode::HU => predict_bhupred(&mut ws, x0, y0, stride),
            }

            let rb: &[i32; 16] = block.try_into().unwrap();
            add_residue(&mut ws, rb, y0, x0, stride);
        }

        let dst_stride = self.frame.mb_width * 16;
        let dst_line_ofs = mbx * 16;
        let src_rows = ws.chunks_exact(stride);
        let dst_rows = self.frame.ybuf.chunks_exact_mut(dst_stride).skip(mby * 16);
        for (dst_row, src_row) in dst_rows.zip(src_rows.skip(1)).take(16) {
            dst_row[dst_line_ofs..dst_line_ofs + 16].copy_from_slice(&src_row[1..1 + 16]);
        }
    }

    fn intra_predict_luma_row_0(
        &mut self,
        mbx: usize,
        mode: I16x16Mode,
    ) -> [u8; I16X16_LUMA_BLOCK_SIZE] {
        let mb_width = self.frame.mb_width;
        let luma = &self.frame.ybuf;

        match mode {
            I16x16Mode::DC if mbx == 0 => predict_luma_dc_row_0_col_0(),
            I16x16Mode::DC => predict_luma_dc_row_0_col_x(mbx, mb_width, luma),
            I16x16Mode::VE => predict_luma_vpred_row_0(),
            I16x16Mode::HZ if mbx == 0 => predict_luma_hpred_col_0(),
            I16x16Mode::HZ => predict_luma_hpred_col_x(mbx, 0, mb_width, luma),
            I16x16Mode::TM if mbx == 0 => predict_luma_hpred_col_0(),
            I16x16Mode::TM => predict_luma_tmpred_row_0(mbx, mb_width, luma),
        }
    }

    fn intra_predict_luma_row_x(
        &mut self,
        mbx: usize,
        mby: usize,
        mode: I16x16Mode,
    ) -> [u8; I16X16_LUMA_BLOCK_SIZE] {
        let mb_width = self.frame.mb_width;
        let luma = &self.frame.ybuf;

        match mode {
            I16x16Mode::DC if mbx == 0 => predict_luma_dc_row_x_col_0(mby, mb_width, luma),
            I16x16Mode::DC => predict_luma_dc_row_x_col_x(mbx, mby, mb_width, luma),
            I16x16Mode::VE => predict_luma_vpred_row_x(mbx, mby, mb_width, luma),
            I16x16Mode::HZ if mbx == 0 => predict_luma_hpred_col_0(),
            I16x16Mode::HZ => predict_luma_hpred_col_x(mbx, mby, mb_width, luma),
            I16x16Mode::TM if mbx == 0 => predict_luma_tmpred_row_x_col_0(mby, mb_width, luma),
            I16x16Mode::TM => predict_luma_tmpred_row_x(mbx, mby, mb_width, luma),
        }
    }

    fn add_residue_luma(
        &mut self,
        mbx: usize,
        mby: usize,
        mut ws: [u8; I16X16_LUMA_BLOCK_SIZE],
        resdata: &[i32],
    ) {
        for (i, block) in resdata.chunks(16).take(16).enumerate() {
            let x = (i % 4) * 4;
            let y = (i / 4) * 4;
            let rb: &[i32; 16] = block.try_into().unwrap();
            add_residue(&mut ws, rb, y, x, I16X16_LUMA_STRIDE);
        }

        let dst_stride = self.frame.mb_width * 16;
        let dst_line_ofs = mbx * 16;
        let src_rows = ws.chunks_exact(I16X16_LUMA_STRIDE);
        let dst_rows = self.frame.ybuf.chunks_exact_mut(dst_stride).skip(mby * 16);
        for (dst_row, src_row) in dst_rows.zip(src_rows).take(16) {
            dst_row[dst_line_ofs..dst_line_ofs + 16].copy_from_slice(&src_row[0..16]);
        }
    }

    fn intra_predict_chroma(&mut self, mbx: usize, mby: usize, mb: &Macroblock, resdata: &[i32]) {
        let stride = 1usize + 8;

        let mw = self.frame.mb_width;

        //8x8 with left top border of 1
        let mut uws = create_border_chroma(mbx, mby, &self.top_border_u, &self.left_border_u);
        let mut vws = create_border_chroma(mbx, mby, &self.top_border_v, &self.left_border_v);

        match mb.chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(&mut uws, 8, stride, mby != 0, mbx != 0);
                predict_dcpred(&mut vws, 8, stride, mby != 0, mbx != 0);
            }
            ChromaMode::V => {
                predict_vpred(&mut uws, 8, 1, 1, stride);
                predict_vpred(&mut vws, 8, 1, 1, stride);
            }
            ChromaMode::H => {
                predict_hpred(&mut uws, 8, 1, 1, stride);
                predict_hpred(&mut vws, 8, 1, 1, stride);
            }
            ChromaMode::TM => {
                predict_tmpred(&mut uws, 8, 1, 1, stride);
                predict_tmpred(&mut vws, 8, 1, 1, stride);
            }
        }

        for y in 0..2 {
            for x in 0..2 {
                let i = x + y * 2;
                let urb: &[i32; 16] = resdata[16 * 16 + i * 16..][..16].try_into().unwrap();

                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;
                add_residue(&mut uws, urb, y0, x0, stride);

                let vrb: &[i32; 16] = resdata[20 * 16 + i * 16..][..16].try_into().unwrap();

                add_residue(&mut vws, vrb, y0, x0, stride);
            }
        }

        set_chroma_border(&mut self.left_border_u, &mut self.top_border_u, &uws, mbx);
        set_chroma_border(&mut self.left_border_v, &mut self.top_border_v, &vws, mbx);

        for y in 0..8 {
            let uv_buf_index = (mby * 8 + y) * mw * 8 + mbx * 8;
            let ws_index = (1 + y) * stride + 1;

            for (((ub, vb), &uw), &vw) in self.frame.ubuf[uv_buf_index..][..8]
                .iter_mut()
                .zip(self.frame.vbuf[uv_buf_index..][..8].iter_mut())
                .zip(uws[ws_index..][..8].iter())
                .zip(vws[ws_index..][..8].iter())
            {
                *ub = uw;
                *vb = vw;
            }
        }
    }

    fn read_coefficients<'a>(
        &mut self,
        block: &mut [i32; 16],
        decoder: &mut ArithmeticDecoder<'a>,
        plane: Plane,
        complexity: usize,
        dcq: i16,
        acq: i16,
    ) -> Result<bool> {
        // perform bounds checks once up front,
        // so that the compiler doesn't have to insert them in the hot loop below
        assert!(complexity <= 2);

        let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
        let probs = &self.token_probs[plane as usize];

        let mut complexity = complexity;
        let mut has_coefficients = false;
        let mut skip = false;

        for i in first_coeff..16 {
            const COEFF_BANDS: [u8; 16] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7];
            let band = COEFF_BANDS[i] as usize;
            let tree = &probs[band][complexity];

            let token = read_with_tree_with_first_node(decoder, tree, tree[skip as usize]) as u8;

            let mut abs_value = i32::from(match token {
                DCT_EOB => break,

                DCT_0 => {
                    skip = true;
                    has_coefficients = true;
                    complexity = 0;
                    continue;
                }

                DCT_1..=DCT_4 => i16::from(token),

                DCT_CAT1..=DCT_CAT6 => {
                    let category = usize::from(token - DCT_CAT1);

                    const PROB_DCT_CAT: [[u8; 12]; 6] = [
                        [159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [165, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [173, 148, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [176, 155, 140, 135, 0, 0, 0, 0, 0, 0, 0, 0],
                        [180, 157, 141, 134, 130, 0, 0, 0, 0, 0, 0, 0],
                        [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0],
                    ];
                    let probs = PROB_DCT_CAT[category];

                    let mut extra = 0i16;

                    for t in probs.iter().copied() {
                        if t == 0 {
                            break;
                        }
                        let b = decoder.read_bool(t);
                        extra = 2 * extra + i16::from(b);
                    }

                    const DCT_CAT_BASE: [u8; 6] = [5, 7, 11, 19, 35, 67];
                    i16::from(DCT_CAT_BASE[category]) + extra
                }

                c => panic!("unknown token: {c}"),
            });

            skip = false;

            complexity = if abs_value == 0 {
                0
            } else if abs_value == 1 {
                1
            } else {
                2
            };

            abs_value = decoder.read_signed(abs_value);
            //if decoder.read_flag() {
            //    abs_value = -abs_value;
            //}

            const ZIGZAG: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
            let zigzag = usize::from(ZIGZAG[i]);
            block[zigzag] = abs_value * i32::from(if zigzag > 0 { acq } else { dcq });

            has_coefficients = true;
        }

        if decoder.is_overflow() {
            Err(Error::BufferUnderrun)
        } else {
            Ok(has_coefficients)
        }
    }

    fn read_residual_data<'a>(
        &mut self,
        mb: &mut Macroblock,
        mbx: usize,
        decoder: &mut ArithmeticDecoder<'a>,
    ) -> Result<[i32; 384]> {
        let sindex = mb.segmentid as usize;
        let mut blocks = [0i32; 384];
        let mut plane = if mb.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        if plane == Plane::Y2 {
            let complexity = self.top[mbx].complexity[0] + self.left.complexity[0];
            let mut block = [0i32; 16];
            let dcq = self.segment[sindex].y2dc;
            let acq = self.segment[sindex].y2ac;
            let n =
                self.read_coefficients(&mut block, decoder, plane, complexity as usize, dcq, acq)?;

            self.left.complexity[0] = if n { 1 } else { 0 };
            self.top[mbx].complexity[0] = if n { 1 } else { 0 };

            vp8_xform::iwht4x4(&mut block);

            for k in 0..16 {
                blocks[16 * k] = block[k];
            }

            plane = Plane::YCoeff1;
        }

        for y in 0..4 {
            let mut left = self.left.complexity[y + 1];
            for x in 0..4 {
                let i = x + y * 4;
                let block = &mut blocks[i * 16..][..16];
                let block: &mut [i32; 16] = block.try_into().unwrap();

                let complexity = self.top[mbx].complexity[x + 1] + left;
                let dcq = self.segment[sindex].ydc;
                let acq = self.segment[sindex].yac;

                let n =
                    self.read_coefficients(block, decoder, plane, complexity as usize, dcq, acq)?;

                if block[0] != 0 || n {
                    mb.non_zero_dct = true;
                    vp8_xform::idct4x4(block);
                }

                left = if n { 1 } else { 0 };
                self.top[mbx].complexity[x + 1] = if n { 1 } else { 0 };
            }

            self.left.complexity[y + 1] = left;
        }

        plane = Plane::Chroma;

        for &j in &[5usize, 7usize] {
            for y in 0..2 {
                let mut left = self.left.complexity[y + j];

                for x in 0..2 {
                    let i = x + y * 2 + if j == 5 { 16 } else { 20 };
                    let block = &mut blocks[i * 16..][..16];
                    let block: &mut [i32; 16] = block.try_into().unwrap();

                    let complexity = self.top[mbx].complexity[x + j] + left;
                    let dcq = self.segment[sindex].uvdc;
                    let acq = self.segment[sindex].uvac;

                    let n = self.read_coefficients(
                        block,
                        decoder,
                        plane,
                        complexity as usize,
                        dcq,
                        acq,
                    )?;
                    if block[0] != 0 || n {
                        mb.non_zero_dct = true;
                        vp8_xform::idct4x4(block);
                    }

                    left = if n { 1 } else { 0 };
                    self.top[mbx].complexity[x + j] = if n { 1 } else { 0 };
                }

                self.left.complexity[y + j] = left;
            }
        }

        Ok(blocks)
    }

    fn loop_filter_mb(&mut self, mbx: usize, mby: usize) {
        let mb = &self.macroblocks[mby * self.frame.mb_width + mbx];

        let luma_w = self.frame.mb_width * 16;
        let chroma_w = self.frame.mb_width * 8;

        let (filter_level, interior_limit, hev_threshold) = self.calculate_filter_parameters(mb);

        if filter_level > 0 {
            let mbedge_limit = (filter_level + 2) * 2 + interior_limit;
            let sub_bedge_limit = (filter_level * 2) + interior_limit;

            // we skip subblock filtering if the coding mode isn't B_PRED and there's no DCT coefficient coded
            let do_subblock_filtering =
                mb.luma_mode == LumaMode::B || (!mb.coeffs_skipped && mb.non_zero_dct);

            //filter across left of macroblock
            if mbx > 0 {
                //simple loop filtering
                if self.frame.filter_type {
                    for y in 0..16 {
                        let y0 = mby * 16 + y;
                        let x0 = mbx * 16;

                        vp8_loop_filter::simple_segment_horizontal(
                            mbedge_limit,
                            &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                        );
                    }
                } else {
                    for y in 0..16 {
                        let y0 = mby * 16 + y;
                        let x0 = mbx * 16;

                        vp8_loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                        );
                    }

                    for y in 0..8 {
                        let y0 = mby * 8 + y;
                        let x0 = mbx * 8;

                        vp8_loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ubuf[y0 * chroma_w + x0 - 4..][..8],
                        );
                        vp8_loop_filter::macroblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.vbuf[y0 * chroma_w + x0 - 4..][..8],
                        );
                    }
                }
            }

            //filter across vertical subblocks in macroblock
            if do_subblock_filtering {
                if self.frame.filter_type {
                    for x in (4usize..16 - 1).step_by(4) {
                        for y in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            vp8_loop_filter::simple_segment_horizontal(
                                sub_bedge_limit,
                                &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                            );
                        }
                    }
                } else {
                    for x in (4usize..16 - 3).step_by(4) {
                        for y in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            vp8_loop_filter::subblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.frame.ybuf[y0 * luma_w + x0 - 4..][..8],
                            );
                        }
                    }

                    for y in 0..8 {
                        let y0 = mby * 8 + y;
                        let x0 = mbx * 8 + 4;

                        vp8_loop_filter::subblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.ubuf[y0 * chroma_w + x0 - 4..][..8],
                        );

                        vp8_loop_filter::subblock_filter_horizontal(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.vbuf[y0 * chroma_w + x0 - 4..][..8],
                        );
                    }
                }
            }

            //filter across top of macroblock
            if mby > 0 {
                if self.frame.filter_type {
                    for x in 0..16 {
                        let y0 = mby * 16;
                        let x0 = mbx * 16 + x;

                        vp8_loop_filter::simple_segment_vertical(
                            mbedge_limit,
                            &mut self.frame.ybuf[..],
                            y0 * luma_w + x0,
                            luma_w,
                        );
                    }
                } else {
                    //if bottom macroblock, can only filter if there is 3 pixels below
                    for x in 0..16 {
                        let y0 = mby * 16;
                        let x0 = mbx * 16 + x;

                        vp8_loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ybuf[..],
                            y0 * luma_w + x0,
                            luma_w,
                        );
                    }

                    for x in 0..8 {
                        let y0 = mby * 8;
                        let x0 = mbx * 8 + x;

                        vp8_loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.ubuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );
                        vp8_loop_filter::macroblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            mbedge_limit,
                            &mut self.frame.vbuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );
                    }
                }
            }

            //filter across horizontal subblock edges within the macroblock
            if do_subblock_filtering {
                if self.frame.filter_type {
                    for y in (4usize..16 - 1).step_by(4) {
                        for x in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            vp8_loop_filter::simple_segment_vertical(
                                sub_bedge_limit,
                                &mut self.frame.ybuf[..],
                                y0 * luma_w + x0,
                                luma_w,
                            );
                        }
                    }
                } else {
                    for y in (4usize..16 - 3).step_by(4) {
                        for x in 0..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16 + x;

                            vp8_loop_filter::subblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.frame.ybuf[..],
                                y0 * luma_w + x0,
                                luma_w,
                            );
                        }
                    }

                    for x in 0..8 {
                        let y0 = mby * 8 + 4;
                        let x0 = mbx * 8 + x;

                        vp8_loop_filter::subblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.ubuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );

                        vp8_loop_filter::subblock_filter_vertical(
                            hev_threshold,
                            interior_limit,
                            sub_bedge_limit,
                            &mut self.frame.vbuf[..],
                            y0 * chroma_w + x0,
                            chroma_w,
                        );
                    }
                }
            }
        }
    }

    //return values are the filter level, interior limit and hev threshold
    fn calculate_filter_parameters(&self, mb: &Macroblock) -> (u8, u8, u8) {
        let segment = self.segment[mb.segmentid as usize];
        let mut filter_level = i32::from(self.frame.filter_level);

        // if frame level filter level is 0, we must skip loop filter
        if filter_level == 0 {
            return (0, 0, 0);
        }

        if self.segments_enabled {
            if segment.delta_values {
                filter_level += i32::from(segment.loopfilter_level);
            } else {
                filter_level = i32::from(segment.loopfilter_level);
            }
        }

        filter_level = filter_level.clamp(0, 63);

        if self.loop_filter_adjustments_enabled {
            filter_level += self.ref_delta[0];
            if mb.luma_mode == LumaMode::B {
                filter_level += self.mode_delta[0];
            }
        }

        let filter_level = filter_level.clamp(0, 63) as u8;

        //interior limit
        let mut interior_limit = filter_level;

        if self.frame.sharpness_level > 0 {
            interior_limit >>= if self.frame.sharpness_level > 4 { 2 } else { 1 };

            if interior_limit > 9 - self.frame.sharpness_level {
                interior_limit = 9 - self.frame.sharpness_level;
            }
        }

        if interior_limit == 0 {
            interior_limit = 1;
        }

        // high edge variance threshold
        let hev_threshold = if filter_level >= 40 {
            2
        } else if filter_level >= 15 {
            1
        } else {
            0
        };

        (filter_level, interior_limit, hev_threshold)
    }

    // ------------------------------------------------------------------------
    fn read_uncompressed_header(&mut self, data: &[u8]) -> Result<()> {
        let header = read_le24(&data[0..3].try_into()?);

        let keyframe = header & 1 == 0;
        if !keyframe {
            return Err(Error::NonKeyframe);
        }

        let for_display = (header >> 4) & 1 != 0;
        let first_partition_size = header >> 5;

        let tag = data[3..6].try_into()?;
        if tag != [0x9d, 0x01, 0x2a] {
            return Err(Error::Vp8MagicInvalid(tag));
        }

        self.frame.width = usize::from(u16::from_le_bytes(data[6..8].try_into()?) & 0x3fff);
        self.frame.height = usize::from(u16::from_le_bytes(data[8..10].try_into()?) & 0x3fff);
        self.frame.mb_width = self.frame.width.div_ceil(16);
        self.frame.mb_height = self.frame.height.div_ceil(16);

        let mb_count = self.frame.mb_width * self.frame.mb_height;
        self.frame.ybuf = vec![0u8; mb_count * 16 * 16];
        self.frame.ubuf = vec![0u8; mb_count * 8 * 8];
        self.frame.vbuf = vec![0u8; mb_count * 8 * 8];

        self.frame.for_display = for_display;
        self.frame.first_partition_size = first_partition_size;

        self.top = vec![MacroblockContext::default(); self.frame.mb_width];
        self.left = self.top.first().copied().unwrap_or_default();

        self.top_border_u = vec![127u8; 8 * self.frame.mb_width];
        self.left_border_u = vec![129u8; 1 + 8];

        self.top_border_v = vec![127u8; 8 * self.frame.mb_width];
        self.left_border_v = vec![129u8; 1 + 8];

        Ok(())
    }

    // ------------------------------------------------------------------------
    fn read_compressed_header<'a>(&mut self, b: &mut ArithmeticDecoder<'a>) -> Result<()> {
        let color_space = b.read_literal(1);
        if color_space != 0 {
            return Err(Error::ColorSpaceInvalid(color_space));
        }

        let _pixel_type = b.read_literal(1);

        self.segments_enabled = b.read_flag();
        if self.segments_enabled {
            self.read_segment_updates(b)?;
        }

        self.frame.filter_type = b.read_flag();
        self.frame.filter_level = b.read_literal(6);
        self.frame.sharpness_level = b.read_literal(3);

        self.loop_filter_adjustments_enabled = b.read_flag();
        if self.loop_filter_adjustments_enabled {
            self.read_loop_filter_adjustments(b)?;
        }

        self.num_partitions = 1 << b.read_literal(2) as usize;

        if b.is_overflow() {
            Err(Error::BufferUnderrun)
        } else {
            Ok(())
        }
    }

    pub fn decode_frame(mut self, data: &[u8]) -> Result<Frame> {
        let (head, partitions) = data.split_at(10);
        self.read_uncompressed_header(head)?;

        let (partition, partitions) = partitions.split_at(self.frame.first_partition_size);
        let mut b = ArithmeticDecoder::new(partition);

        self.read_compressed_header(&mut b)?;

        let mut decoders = Vec::new();
        let (sizes, tail) = partitions.split_at(3 * (self.num_partitions - 1));
        let (sizes, _) = sizes.as_chunks::<3>();

        let mut remain = tail;
        for s in sizes {
            let size = read_le24(s);

            let (data, tail) = remain.split_at(size);
            decoders.push(ArithmeticDecoder::new(data));

            remain = tail;
        }

        decoders.push(ArithmeticDecoder::new(remain));

        self.read_quantization_indices(&mut b)?;

        // Refresh entropy probs ?????
        let _ = b.read_literal(1);

        self.update_token_probabilities(&mut b)?;

        let mb_no_skip_coeff = b.read_literal(1);
        self.prob_skip_false = if mb_no_skip_coeff == 1 {
            Some(b.read_literal(8))
        } else {
            None
        };
        if b.is_overflow() {
            return Err(Error::BufferUnderrun);
        }

        {
            self.left = MacroblockContext::default();
            for mbx in 0..self.frame.mb_width {
                let (mut mb, pred) = self.read_macroblock_header(&mut b, mbx)?;
                let blocks = if !mb.coeffs_skipped {
                    self.read_residual_data(&mut mb, mbx, &mut decoders[0])?
                } else {
                    if mb.luma_mode != LumaMode::B {
                        self.left.complexity[0] = 0;
                        self.top[mbx].complexity[0] = 0;
                    }

                    for i in 1usize..9 {
                        self.left.complexity[i] = 0;
                        self.top[mbx].complexity[i] = 0;
                    }

                    [0i32; 384]
                };

                match pred {
                    PredictionType::I4x4(modes) => {
                        self.intra_predict_luma_b_row_0(mbx, modes, &blocks)
                    }
                    PredictionType::I16x16(mode) => {
                        let ws = self.intra_predict_luma_row_0(mbx, mode);
                        self.add_residue_luma(mbx, 0, ws, &blocks);
                    }
                }
                self.intra_predict_chroma(mbx, 0, &mb, &blocks);

                self.macroblocks.push(mb);
            }
        }

        self.left_border_u = vec![129u8; 1 + 8];
        self.left_border_v = vec![129u8; 1 + 8];

        for mby in 1..self.frame.mb_height {
            let p = mby % self.num_partitions;
            let decoder = &mut decoders[p];
            self.left = MacroblockContext::default();

            for mbx in 0..self.frame.mb_width {
                let (mut mb, pred) = self.read_macroblock_header(&mut b, mbx)?;
                let blocks = if !mb.coeffs_skipped {
                    self.read_residual_data(&mut mb, mbx, decoder)?
                } else {
                    if mb.luma_mode != LumaMode::B {
                        self.left.complexity[0] = 0;
                        self.top[mbx].complexity[0] = 0;
                    }

                    for i in 1usize..9 {
                        self.left.complexity[i] = 0;
                        self.top[mbx].complexity[i] = 0;
                    }

                    [0i32; 384]
                };

                match pred {
                    PredictionType::I4x4(modes) => {
                        self.intra_predict_luma_b_row_x(mbx, mby, modes, &blocks)
                    }
                    PredictionType::I16x16(mode) => {
                        let ws = self.intra_predict_luma_row_x(mbx, mby, mode);
                        self.add_residue_luma(mbx, mby, ws, &blocks);
                    }
                }
                self.intra_predict_chroma(mbx, mby, &mb, &blocks);

                self.macroblocks.push(mb);
            }

            self.left_border_u = vec![129u8; 1 + 8];
            self.left_border_v = vec![129u8; 1 + 8];
        }

        for mby in 0..self.frame.mb_height {
            for mbx in 0..self.frame.mb_width {
                self.loop_filter_mb(mbx, mby);
            }
        }

        println!(
            "Intra4x4 count: {}, Intra16x16 count: {}",
            self.intra4x4_count, self.intra16x16_count
        );

        Ok(self.frame)
    }
}

// set border
fn set_chroma_border(
    left_border: &mut [u8],
    top_border: &mut [u8],
    chroma_block: &[u8],
    mbx: usize,
) {
    let stride = 1usize + 8;
    // top left is top right of previous chroma block
    left_border[0] = chroma_block[8];

    // left border
    for (i, left) in left_border[1..][..8].iter_mut().enumerate() {
        *left = chroma_block[(i + 1) * stride + 8];
    }

    for (top, &w) in top_border[mbx * 8..][..8]
        .iter_mut()
        .zip(&chroma_block[8 * stride + 1..][..8])
    {
        *top = w;
    }
}
