#![recursion_limit = "256"]

use burn::backend::Wgpu;
use burn::tensor::Tensor;

extern crate guide;

type Backend = Wgpu;

fn main() {
    // let device = Default::default();

    // let tensor1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);

    // let tensor2 = Tensor::<Backend, 2>::ones_like(&tensor1);

    // println!("{}", tensor1 + tensor2);
    guide::guide();
}
