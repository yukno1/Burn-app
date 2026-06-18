use burn::{
    backend::{Autodiff, Wgpu},
    config::Config,
    nn::{DropoutConfig, LinearConfig, Relu, conv::Conv2dConfig, pool::AdaptiveAvgPool2dConfig},
    optim::AdamConfig,
    tensor::{Device, backend::Backend},
};

use crate::{model::Model, training::TrainingConfig};

mod data;
mod model;
mod training;

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
        }
    }
}

pub fn guide() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutoDiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = ".\\tmp\\guide";

    crate::training::train::<MyAutoDiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
