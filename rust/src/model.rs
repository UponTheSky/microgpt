use std::{collections::HashMap, hash::Hash, iter::zip};

use crate::{autograd::ValueRef, parameters::Parameters};

pub struct Model {
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    head_dim: usize,
    pub parameters: Parameters,
    keys: Vec<Vec<Vec<ValueRef>>>,   // KV Cache
    values: Vec<Vec<Vec<ValueRef>>>, // KV Cache
}

impl Model {
    pub fn new(
        n_layer: usize,
        n_head: usize,
        n_embd: usize,
        parameters: Parameters,
        keys: Vec<Vec<Vec<ValueRef>>>,
        values: Vec<Vec<Vec<ValueRef>>>,
    ) -> Self {
        Model {
            n_layer,
            n_head,
            n_embd,
            head_dim: n_embd / n_head,
            parameters,
            keys,
            values,
        }
    }

    pub fn reset_cache(&mut self) {
        self.keys.clear();
        self.keys.push(Vec::new());

        self.values.clear();
        self.values.push(Vec::new());
    }

    pub fn gpt(&mut self, token_id: usize, pos_id: usize) -> Vec<ValueRef> {
        let tok_emb = self
            .parameters
            .state_dict
            .get("wte")
            .unwrap()
            .get(token_id)
            .unwrap();
        let pos_emb = self
            .parameters
            .state_dict
            .get("wpe")
            .unwrap()
            .get(pos_id)
            .unwrap();

        let x = zip(tok_emb, pos_emb)
            .map(|(t, p)| t.clone() + p.clone())
            .collect();

        let mut x = rmsnorm(&x);
        let mut x_residual: Vec<ValueRef> = Vec::new();

        for li in 0..self.n_layer {
            let new_x = rmsnorm(&x);
            x_residual = x;
            x = new_x;

            let q = linear(
                &x,
                self.parameters
                    .state_dict
                    .get(format!("layer{}.attn_wq", li).as_str())
                    .unwrap(),
            );
            let k = linear(
                &x,
                self.parameters
                    .state_dict
                    .get(format!("layer{}.attn_wk", li).as_str())
                    .unwrap(),
            );
            let v = linear(
                &x,
                self.parameters
                    .state_dict
                    .get(format!("layer{}.attn_wv", li).as_str())
                    .unwrap(),
            );

            let li_keys = self.keys.get_mut(li).unwrap();
            let li_values = self.values.get_mut(li).unwrap();

            li_keys.push(k);
            li_values.push(v);

            let mut x_attn = Vec::new();

            for h in 0..self.n_head {
                let hs = h * self.head_dim;
                let q_h = &q[hs..(hs + self.head_dim)];
                let k_h: Vec<&[ValueRef]> = li_keys
                    .iter()
                    .map(|key| &key[hs..(hs + self.head_dim)])
                    .collect();
                let v_h: Vec<&[ValueRef]> = li_values
                    .iter()
                    .map(|value| &value[hs..(hs + self.head_dim)])
                    .collect();

                let attn_logits: Vec<ValueRef> = k_h
                    .into_iter()
                    .map(|key| {
                        zip(q_h, key).fold(ValueRef::new_with_defaults(0.0), |acc, (a, b)| {
                            acc + (a.clone() * b.clone())
                        }) / ValueRef::new_with_defaults(self.head_dim as f64)
                            .pow(ValueRef::new_with_defaults(0.5))
                    })
                    .collect();

                let attn_weights = softmax(&attn_logits);

                // this part has different matrix multiplication direction: weight * matrix
                let head_out = (0..self.head_dim).map(|j| {
                    (0..v_h.len()).fold(ValueRef::new_with_defaults(0.0), |acc, t| {
                        acc + (attn_weights[t].clone() * v_h[t][j].clone())
                    })
                });

                x_attn.extend(head_out);
            }

            x = linear(
                &x_attn,
                self.parameters
                    .state_dict
                    .get(format!("layer{}.attn_wo", li).as_str())
                    .unwrap(),
            );
            x = zip(&x, x_residual)
                .map(|(a, b)| a.clone() + b.clone())
                .collect();

            let new_x = rmsnorm(&x);
            x_residual = x;
            x = new_x;

            x = linear(
                &x,
                self.parameters
                    .state_dict
                    .get(format!("layer{li}.mlp_fc1").as_str())
                    .unwrap(),
            );
            x = x.iter().map(|el| el.clone().relu()).collect();
            x = linear(
                &x,
                self.parameters
                    .state_dict
                    .get(format!("layer{li}.mlp_fc2").as_str())
                    .unwrap(),
            );
            x = zip(&x, x_residual)
                .map(|(a, b)| a.clone() + b.clone())
                .collect();
        }

        linear(&x, self.parameters.state_dict.get("lm_head").unwrap())
    }
}

fn linear(x: &Vec<ValueRef>, w: &Vec<Vec<ValueRef>>) -> Vec<ValueRef> {
    w.iter()
        .map(|row| {
            zip(row, x).fold(ValueRef::new_with_defaults(0.0), |acc, (a, b)| {
                acc + (a.clone() * b.clone())
            })
        })
        .collect()
}

pub fn softmax(logits: &Vec<ValueRef>) -> Vec<ValueRef> {
    let max_val = logits
        .iter()
        .max_by(|x, y| f64::total_cmp(&x.data(), &y.data()))
        .unwrap();
    let exps: Vec<ValueRef> = logits
        .iter()
        .map(|val| (val.clone() - max_val.clone()).exp())
        .collect();
    let total = exps
        .iter()
        .map(|x| x.clone())
        .fold(ValueRef::new_with_defaults(0.0), |acc, b| acc + b);

    exps.into_iter().map(|e| e / total.clone()).collect()
}

fn rmsnorm(x: &Vec<ValueRef>) -> Vec<ValueRef> {
    let ms = x.iter().fold(ValueRef::new_with_defaults(0.0), |acc, val| {
        acc + (val.clone() * val.clone())
    }) / ValueRef::new_with_defaults(x.len() as f64);
    let scale = (ms + ValueRef::new_with_defaults(1e-5)).pow(ValueRef::new_with_defaults(-0.5));

    x.iter().map(|val| val.clone() * scale.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let v: Vec<ValueRef> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| ValueRef::new_with_defaults(v))
            .collect();

        let w0: Vec<ValueRef> = vec![1.0, 1.0, 2.0]
            .into_iter()
            .map(|v| ValueRef::new_with_defaults(v))
            .collect();

        let w1: Vec<ValueRef> = vec![2.0, 5.0, 4.0]
            .into_iter()
            .map(|v| ValueRef::new_with_defaults(v))
            .collect();

        let lineared = linear(&v, &vec![w0, w1]);
        let expected = vec![9.0, 24.0];

        assert_eq!(
            expected,
            lineared.into_iter().map(|v| v.data()).collect::<Vec<f64>>()
        );
    }

    #[test]
    fn test_softmax() {
        let v: Vec<ValueRef> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| ValueRef::new_with_defaults(v))
            .collect();

        let softmaxed = softmax(&v);

        for val in softmaxed {
            assert!(val.data() <= 1.0);
            assert!(val.data() >= 0.0);
        }
    }
}
