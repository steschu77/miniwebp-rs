#![feature(test)]
extern crate test;

use test::Bencher;

fn test_bench(b: &mut Bencher, file: &str) {
    let contents = std::fs::read(format!("tests/images/{file}.webp")).unwrap();
    b.iter(|| {
        miniwebp::read_image(&contents).unwrap();
    });
}

#[bench]
fn test_img1(b: &mut Bencher) {
    test_bench(b, "img1");
}

#[bench]
fn test_img2(b: &mut Bencher) {
    test_bench(b, "img2");
}

#[bench]
fn test_img3(b: &mut Bencher) {
    test_bench(b, "img3");
}

#[bench]
fn test_img4(b: &mut Bencher) {
    test_bench(b, "img4");
}

#[bench]
fn test_img5(b: &mut Bencher) {
    test_bench(b, "img5");
}

//#[bench]
fn test_img6(b: &mut Bencher) {
    test_bench(b, "img6");
}

//#[bench]
fn test_img7(b: &mut Bencher) {
    test_bench(b, "img7");
}
