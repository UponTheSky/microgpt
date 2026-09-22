pub enum Token {
    C(char), // a ~ z
    Bos,     // BOS
}

pub type TokenId = usize;

const LOWER_A_ASCII_VALUE: u8 = 'a' as u8;
const BOS: &'static str = "BOS";

pub struct Tokenizer {}

impl Tokenizer {
    pub fn vocabulary(token: Token) -> Option<TokenId> {
        match token {
            Token::C(ch) => {
                if !ch.is_alphabetic() {
                    return None;
                }

                Some((ch as u8 - LOWER_A_ASCII_VALUE) as usize)
            }
            Token::Bos => {
                Some(26usize) // a ~ z + 1
            }
        }
    }

    pub fn to_char(token_id: TokenId) -> char {
        (token_id as u8 + LOWER_A_ASCII_VALUE) as char
    }
}
