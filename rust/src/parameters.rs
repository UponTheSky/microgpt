use std::collections::HashMap;

use crate::autograd::ValueRef;
use rand::{self, rand_core::block};
use rand_distr::{Distribution, Normal};

pub struct Parameters {
    // manually set
    pub n_embd: usize,     // embedding dimension
    pub n_head: usize,     // number of the heads
    pub n_layer: usize,    // number of the layers
    pub block_size: usize, // the context window length

    init_std: f64,

    // derived
    pub head_dim: usize, // head dimension
    pub state_dict: HashMap<String, Vec<Vec<ValueRef>>>,
}

impl Parameters {
    pub fn new(
        vocab_size: usize,
        n_embd: usize,
        n_head: usize,
        n_layer: usize,
        block_size: usize,
        init_std: f64,
    ) -> Self {
        let mut state_dict = HashMap::new();

        // note that wte and wpe are not matrix - more like a dictionary that returns a vector for a given integer id
        state_dict.insert(
            "wte".into(),
            Parameters::matrix(vocab_size, n_embd, init_std),
        ); // token embedding
        state_dict.insert(
            "wpe".into(),
            Parameters::matrix(block_size, n_embd, init_std),
        ); // positional embedding
        state_dict.insert(
            "lm_head".into(),
            Parameters::matrix(vocab_size, n_embd, init_std),
        ); // positional embedding

        for i in (0..n_layer) {
            state_dict.insert(
                format!("layer{}.attn_wq", i),
                Parameters::matrix(n_embd, n_embd, init_std),
            );
            state_dict.insert(
                format!("layer{}.attn_wk", i),
                Parameters::matrix(n_embd, n_embd, init_std),
            );
            state_dict.insert(
                format!("layer{}.attn_wv", i),
                Parameters::matrix(n_embd, n_embd, init_std),
            );
            state_dict.insert(
                format!("layer{}.attn_wo", i),
                Parameters::matrix(n_embd, n_embd, init_std),
            );
            state_dict.insert(
                format!("layer{}.mlp_fc1", i),
                Parameters::matrix(4 * n_embd, n_embd, init_std),
            );
            state_dict.insert(
                format!("layer{}.mlp_fc2", i),
                Parameters::matrix(n_embd, 4 * n_embd, init_std),
            );
        }

        Self {
            n_embd,
            n_head,
            n_layer,
            block_size,
            init_std,
            head_dim: n_embd / n_head,
            state_dict,
        }
    }

    fn matrix(nout: usize, nin: usize, std: f64) -> Vec<Vec<ValueRef>> {
        let normal = Normal::new(0., std).unwrap();

        (0..nout)
            .map(|_| {
                (0..nin)
                    .map(|_| ValueRef::new_with_defaults(normal.sample(&mut rand::rng())))
                    .collect()
            })
            .collect::<Vec<Vec<ValueRef>>>()
    }

    fn params(&self) -> Vec<ValueRef> {
        self.state_dict
            .values()
            .flatten()
            .flatten()
            .map(|value| value.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_params() {
        // asset
        let params = Parameters::new(4, 8, 2, 2, 4, 1.);

        // action
        let params_count = params.params().len();

        // assert
        assert_eq!(params_count, 32 + 32 + 32 + (4 * 8 * 8 + 2 * 4 * 8 * 8) * 2);
    }
}
