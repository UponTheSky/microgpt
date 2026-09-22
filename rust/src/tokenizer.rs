pub enum Token {
    C(char), // a ~ z
    Bos,     // BOS
}

pub type TokenId = u32;

const LOWER_A_ASCII_VALUE: u32 = 'a' as u32;
const BOS: &'static str = "BOS";

pub struct Tokenizer {}

impl Tokenizer {
    pub fn vocabulary(token: Token) -> Option<TokenId> {
        match token {
            Token::C(ch) => {
                if !ch.is_alphabetic() {
                    return None;
                }

                Some(ch as u32 - LOWER_A_ASCII_VALUE)
            }
            Token::Bos => {
                Some(26u32) // a ~ z + 1
            }
        }
    }
}
