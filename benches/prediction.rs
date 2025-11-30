#![feature(test)]
extern crate test;

use miniwebp::vp8_ipred::*;
use test::Bencher;

const W: usize = 256;
const H: usize = 256;

fn make_sample_image() -> Vec<u8> {
    let mut v = Vec::with_capacity(W * H * 4);
    for c in 0u8..=255 {
        for k in 0u8..=255 {
            v.push(c);
            v.push(0);
            v.push(0);
            v.push(k);
        }
    }
    v
}

#[bench]
fn bench_predict_4x4(b: &mut Bencher) {
    let mut v = make_sample_image();

    b.iter(|| {
        predict_tmpred(&mut v, W * 2, 0, 0, LUMA_STRIDE);
    });
}

#[bench]
fn bench_predict_bvepred(b: &mut Bencher) {
    let mut v = make_sample_image();

    b.iter(|| {
        predict_bvepred(&mut v, 5, 5, W * 2);
    });
}

#[bench]
fn bench_predict_bldpred(b: &mut Bencher) {
    let mut v = make_sample_image();

    b.iter(|| {
        predict_bldpred(&mut v, 5, 5, W * 2);
    });
}

#[bench]
fn bench_predict_brdpred(b: &mut Bencher) {
    let mut v = make_sample_image();

    b.iter(|| {
        predict_brdpred(&mut v, 5, 5, W * 2);
    });
}

#[bench]
fn bench_predict_bhepred(b: &mut Bencher) {
    let mut v = make_sample_image();

    b.iter(|| {
        predict_bhepred(&mut v, 5, 5, W * 2);
    });
}

#[bench]
fn bench_top_pixels(b: &mut Bencher) {
    let v = make_sample_image();

    b.iter(|| {
        top_pixels(&v, 5, 5, W * 2);
    });
}

#[bench]
fn bench_edge_pixels(b: &mut Bencher) {
    let v = make_sample_image();

    b.iter(|| {
        edge_pixels(&v, 5, 5, W * 2);
    });
}
