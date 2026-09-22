use crate::{
    autograd::ValueRef,
    model::{Model, softmax},
    tokenizer::{Token, TokenId, Tokenizer},
};

pub struct TrainConfig {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    eps_adam: f64,
    num_steps: usize,
    block_size: usize,
}

pub fn train(config: TrainConfig, docs: Vec<String>, model: &mut Model) {
    let params = model.parameters.params();
    let params_count = params.len();

    let mut m: Vec<f64> = Vec::with_capacity(params_count);
    let mut v: Vec<f64> = Vec::with_capacity(params_count);

    for step in 0..config.num_steps {
        let doc = docs.get(step % docs.len()).unwrap();
        let doc_token = doc.chars().into_iter().map(|c| Token::C(c));

        let mut token = Vec::new();
        token.push(Token::Bos);
        token.extend(doc_token);
        token.push(Token::Bos);

        let token_ids: Vec<TokenId> = token
            .into_iter()
            .map(|t| Tokenizer::vocabulary(t).unwrap())
            .collect();

        let n = usize::min(config.block_size, token_ids.len() - 1);

        let mut losses: Vec<ValueRef> = Vec::new();

        for pos_id in 0..n {
            let token_id = token_ids.get(pos_id).unwrap().clone() as usize;
            let target_id = token_ids.get(pos_id + 1).unwrap().clone() as usize;
            let logits = model.gpt(token_id, pos_id);

            let probs = softmax(&logits);
            let loss_t = -probs.get(target_id).unwrap().clone();
            losses.push(loss_t);
        }

        let loss = ValueRef::new_with_defaults(1.0 / (n as f64))
            * losses
                .into_iter()
                .fold(ValueRef::new_with_defaults(0.0), |acc, l| acc + l);

        loss.clone().backward();

        let lr_t = config.learning_rate * (1.0 - (step as f64 / config.num_steps as f64));

        for i in 0..params_count {
            let p = params.get(i).unwrap();
            m[i] = config.beta1 * m[i] + (1.0 - config.beta1) * p.grad();
            v[i] = config.beta2 * v[i] + (1.0 - config.beta2) * p.grad() * p.grad();
            let m_hat = m[i] / (1.0 - config.beta1.powi(step as i32 + 1));
            let v_hat = v[i] / (1.0 - config.beta2.powi(step as i32 + 1));

            p.update_data(p.data() - lr_t * m_hat / (v_hat.powf(0.5) + config.eps_adam));
            p.update_grad(0.0);
        }

        eprintln!(
            "step {} / {} | loss {}",
            step + 1,
            config.num_steps,
            loss.data()
        )
    }
}
