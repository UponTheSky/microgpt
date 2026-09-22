use std::{fs, slice::GetDisjointMutError::IndexOutOfBounds};

use crate::{
    autograd::ValueRef, inference::InferenceConfig, model::Model, parameters::Parameters,
    train::TrainConfig,
};

mod autograd;
mod inference;
mod model;
mod parameters;
mod tokenizer;
mod train;

fn main() {
    let dataset_path = std::env::var("DATASET_PATH").unwrap();

    // train
    let docs: Vec<String> = fs::read_to_string(dataset_path)
        .unwrap()
        .split("\n")
        .map(|s| s.into())
        .collect();

    let train_config = TrainConfig {
        learning_rate: 0.01,
        beta1: 0.85,
        beta2: 0.99,
        eps_adam: 1e-8,
        num_steps: 1000,
        block_size: 16,
    };

    let parameters = Parameters::new(26 + 1, 16, 4, 1, 16, 0.08);
    let mut keys: Vec<Vec<Vec<ValueRef>>> = Vec::with_capacity(parameters.n_layer);
    let mut values: Vec<Vec<Vec<ValueRef>>> = Vec::with_capacity(parameters.n_layer);

    keys.push(Vec::new());
    values.push(Vec::new());

    let mut model = Model::new(1, 4, 16, parameters, keys, values);

    train::train(train_config, docs, &mut model);

    // inference
    let infer_config = InferenceConfig {
        temperature: 0.5,
        block_size: 16,
        vocab_size: 26 + 1,
    };

    inference::inference(&mut model, infer_config, 20);
}
