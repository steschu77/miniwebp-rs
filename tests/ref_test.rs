use miniz::zip_read::{zip_open, zip_read};
use std::io::{Cursor, Read, Write};

// ----------------------------------------------------------------------------
fn write_yuv420_file<P: AsRef<std::path::Path>>(
    path: P,
    width: usize,
    height: usize,
    ybuf: &[u8],
    ubuf: &[u8],
    vbuf: &[u8],
) {
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let mb_count = mb_width * mb_height;
    assert_eq!(ybuf.len(), mb_count * 16 * 16);
    assert_eq!(ubuf.len(), mb_count * 8 * 8);
    assert_eq!(vbuf.len(), mb_count * 8 * 8);

    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(ybuf).unwrap();
    file.write_all(ubuf).unwrap();
    file.write_all(vbuf).unwrap();
}

// ----------------------------------------------------------------------------
pub fn read_yuv420_file(data: &[u8], width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mb_width = width.div_ceil(16);
    let mb_height = height.div_ceil(16);
    let mb_count = mb_width * mb_height;

    let mut cursor = Cursor::new(data);

    let mut y = vec![0u8; mb_count * 16 * 16];
    let mut u = vec![0u8; mb_count * 8 * 8];
    let mut v = vec![0u8; mb_count * 8 * 8];

    cursor.read_exact(&mut y).unwrap();
    cursor.read_exact(&mut u).unwrap();
    cursor.read_exact(&mut v).unwrap();

    (y, u, v)
}

// ----------------------------------------------------------------------------
// Read reference YUV
fn read_yuv420_image<P: AsRef<std::path::Path>>(
    file: P,
) -> (usize, usize, Vec<u8>, Vec<u8>, Vec<u8>) {
    let data = std::fs::read(file).unwrap();
    let zip = zip_open(&data).unwrap();

    let meta = zip_read(&data, &zip, "raw.json").unwrap();
    let yuv = zip_read(&data, &zip, "raw.yuv").unwrap();

    let meta = serde_json::from_slice::<serde_json::Value>(&meta).unwrap();
    let width = meta["width"].as_u64().unwrap() as usize;
    let height = meta["height"].as_u64().unwrap() as usize;

    let (y, u, v) = read_yuv420_file(&yuv, width, height);
    (width, height, y, u, v)
}

// ----------------------------------------------------------------------------
fn reference_test(file: &str) {
    let contents = std::fs::read(format!("tests/images/{file}.webp")).unwrap();
    let frame = miniwebp::read_image(&contents).unwrap();

    let (width, height, y, u, v) = read_yuv420_image(format!("tests/reference/{file}.zip"));
    assert_eq!(width, frame.width as usize);
    assert_eq!(height, frame.height as usize);

    // Compare pixels
    let num_diff_luma = frame
        .ybuf
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(num_diff_luma, 0, "Luma pixel mismatch");
    let num_diff_cb = frame
        .ubuf
        .iter()
        .zip(u.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(num_diff_cb, 0, "Chroma blue pixel mismatch");
    let num_diff_cr = frame
        .vbuf
        .iter()
        .zip(v.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(num_diff_cr, 0, "Chroma red pixel mismatch");

    if num_diff_luma > 0 || num_diff_cb > 0 || num_diff_cr > 0 {
        write_yuv420_file(
            format!("tests/test/{file}.{width}x{height}.yuv"),
            width,
            height,
            &frame.ybuf,
            &frame.ubuf,
            &frame.vbuf,
        );
    }
}

// ----------------------------------------------------------------------------
#[test]
fn test_img1() {
    reference_test("img1");
}

#[test]
fn test_img2() {
    reference_test("img2");
}

#[test]
fn test_img3() {
    reference_test("img3");
}

#[test]
fn test_img4() {
    reference_test("img4");
}

#[test]
fn test_img5() {
    reference_test("img5");
}
