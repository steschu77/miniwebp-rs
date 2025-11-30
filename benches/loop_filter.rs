#![feature(test)]
extern crate test;

use miniwebp::vp8_loop_filter::*;
use test::Bencher;

#[rustfmt::skip]
const TEST_DATA: [u8; 8 * 8] = [
    177, 192, 179, 181, 185, 174, 186, 193,
    185, 180, 175, 179, 175, 190, 189, 190,
    185, 181, 177, 190, 190, 174, 176, 188,
    192, 179, 186, 175, 190, 184, 190, 175,
    175, 183, 183, 190, 187, 186, 176, 181,
    183, 177, 182, 185, 183, 179, 178, 181,
    191, 183, 188, 181, 180, 193, 185, 180,
    177, 182, 177, 178, 179, 178, 191, 178,
];

#[bench]
fn measure_horizontal_macroblock_filter(b: &mut Bencher) {
    let hev_threshold = 5;
    let interior_limit = 15;
    let edge_limit = 15;

    let mut data = TEST_DATA;
    let stride = 8;

    b.iter(|| {
        for y in 0..8 {
            macroblock_filter_horizontal(
                hev_threshold,
                interior_limit,
                edge_limit,
                &mut data[y * stride..][..8],
            );
        }
    });
}

#[bench]
fn measure_vertical_macroblock_filter(b: &mut Bencher) {
    let hev_threshold = 5;
    let interior_limit = 15;
    let edge_limit = 15;

    let mut data = TEST_DATA;
    let stride = 8;

    b.iter(|| {
        for x in 0..8 {
            macroblock_filter_vertical(
                hev_threshold,
                interior_limit,
                edge_limit,
                &mut data,
                4 * stride + x,
                stride,
            );
        }
    });
}

#[bench]
fn measure_horizontal_subblock_filter(b: &mut Bencher) {
    let hev_threshold = 5;
    let interior_limit = 15;
    let edge_limit = 15;

    let mut data = TEST_DATA;
    let stride = 8;

    b.iter(|| {
        for y in 0..8 {
            subblock_filter_horizontal(
                hev_threshold,
                interior_limit,
                edge_limit,
                &mut data[y * stride..][..8],
            )
        }
    });
}

#[bench]
fn measure_vertical_subblock_filter(b: &mut Bencher) {
    let hev_threshold = 5;
    let interior_limit = 15;
    let edge_limit = 15;

    let mut data = TEST_DATA;
    let stride = 8;

    b.iter(|| {
        for x in 0..8 {
            subblock_filter_vertical(
                hev_threshold,
                interior_limit,
                edge_limit,
                &mut data,
                4 * stride + x,
                stride,
            )
        }
    });
}

#[bench]
fn measure_simple_segment_horizontal_filter(b: &mut Bencher) {
    let edge_limit = 15;

    let mut data = TEST_DATA;
    let stride = 8;

    b.iter(|| {
        for y in 0..8 {
            simple_segment_horizontal(edge_limit, &mut data[y * stride..][..8])
        }
    });
}

#[bench]
fn measure_simple_segment_vertical_filter(b: &mut Bencher) {
    let edge_limit = 15;

    let mut data = TEST_DATA;
    let stride = 8;

    b.iter(|| {
        for x in 0..16 {
            simple_segment_vertical(edge_limit, &mut data, 4 * stride + x, stride)
        }
    });
}
