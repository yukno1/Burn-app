#![recursion_limit = "256"]

use burn::backend::Wgpu;

extern crate guide;

type Backend = Wgpu;

fn main() {
    guide::guide();
}
