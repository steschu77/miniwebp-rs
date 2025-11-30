# Minimalistic WebP Image Decoding Rust Library
![Rust Workflow](https://github.com/steschu77/miniwebp-rs/actions/workflows/ci.yml/badge.svg)

MiniWebP is a minimalistic decompression library that provides support for decoding lossy VP8 compressed WebP images.

MiniWebP is a fork of the fantastic [image-webp](https://github.com/image-rs/image-webp) library, stripped down to only include the VP8 decompression functionality. This makes MiniWebP a lightweight alternative for projects that only require basic WebP decoding capabilities without the overhead of additional features.

## Project Goals

MiniWebP is intended to be used in projects that require minimal dependencies and a small footprint that need to decode lossy compressed WebP images.

MiniWebP strives to provide fast decompression performance while maintaining a simple and easy-to-use API.

This makes MiniWebP suitable for applications where resource constraints are a concern, such as embedded systems or performance-critical applications.

Non-goals are support for encoding in general, support for lossless WebP images, animation, alpha channels, or color space conversion. If those features are required in your project, consider using [image-webp](https://github.com/image-rs/image-webp) instead.

## Features

* Fast decoding of 'VP8 ' compressed chunks in WebP images
* No dependencies

## Usage

Add the following to your `Cargo.toml`:

```toml
[dependencies]
miniwebp = { git = "https://github.com/steschu77/miniwebp-rs.git" }
```

In your Rust code, you can use the library like this:

```rust
fn main() {
    let contents = std::fs::read(format!("image.webp")).unwrap();
    let frame = miniwebp::read_image(&contents).unwrap();

    // frame contains the decoded YUV image with width, height and pixel data
}
```

## History

This library has been forked from [image-webp](https://github.com/image-rs/image-webp) to provide a more lightweight solution for projects that only require VP8 decoding functionality.

After advanced features had been removed, the code was restructured to make optimizations easier to implement.

Optimizations that are compatible with the original code have been contributed back to the original repository where possible.

## License

This project is licensed under the same licenses as the original [image-webp](https://github.com/image-rs/image-webp) library:

* MIT License
* Apache License 2.0
