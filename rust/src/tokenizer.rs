pub enum Token {
    c(char),           // a ~ z
    bos(&'static str), // BOS
}

pub type TokenId = u32;

const LOWER_A_ASCII_VALUE: u32 = 'a' as u32;
const BOS: &'static str = "BOS";

pub struct Tokenizer {}

impl Tokenizer {
    fn vocabulary(token: Token) -> Option<TokenId> {
        match token {
            Token::c(ch) => {
                if !ch.is_alphabetic() {
                    return None;
                }

                Some(ch as u32 - LOWER_A_ASCII_VALUE)
            }
            Token::bos(bo) => {
                if bo != BOS {
                    return None;
                }

                Some(26u32) // a ~ z + 1
            }
        }
    }
}
