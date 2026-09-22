use rand::RngExt;
use rand_distr::weighted::WeightedIndex;

use crate::{
    autograd::ValueRef,
    model::{Model, softmax},
    tokenizer::{Token, Tokenizer},
};

pub struct InferenceConfig {
    pub temperature: f64,
    pub block_size: usize,
    pub vocab_size: usize,
}

pub fn inference(model: &mut Model, config: InferenceConfig, id_range: u32) {
    eprintln!("\n--- inference (new, hallucinated names) ---");
    let mut rng = rand::rng();

    for sample_idx in 0..id_range {
        let mut token_id = Tokenizer::vocabulary(Token::Bos).unwrap();
        let mut sample = Vec::new();

        for pos_id in 0..config.block_size {
            let logits = model.gpt(token_id, pos_id);
            let probs = softmax(
                &logits
                    .into_iter()
                    .map(|l| l / ValueRef::new_with_defaults(config.temperature))
                    .collect(),
            )
            .into_iter()
            .map(|l| l.data());

            token_id = rng.sample(WeightedIndex::new(probs).unwrap());

            if token_id == Tokenizer::vocabulary(Token::Bos).unwrap() {
                break;
            }

            sample.push(Tokenizer::to_char(token_id));
        }

        eprintln!(
            "sample {}: {}",
            sample_idx + 1,
            sample.iter().collect::<String>()
        );
    }
}
