//! Symbols for Curium markup. Allow for use of Unicode when desired with ASCII fallback.

use crate::markup::{Block, Symbol};

macro_rules! sym {
    ($i:ident, $asc:literal, $uni:literal) => {
        pub const $i: Block = Block::new_symbol($asc, $uni);
    };
}

sym!(SPACE, " ", " ");
sym!(FRAC_SLASH, "/", "\u{2044}");
sym!(MINUS_SIGN, "-", "－");
sym!(PLUS_SIGN, "+", "＋");
sym!(X, "x", "\u{1D465}");
sym!(Y, "y", "\u{1D466}");
sym!(Z, "z", "\u{1D467}");
sym!(LPAREN, "(", "(");
sym!(RPAREN, ")", ")");
sym!(LANGLE, "<", "⟨");
sym!(RANGLE, ">", "⟩");
sym!(A_GLIDE, "a", "a");
sym!(B_GLIDE, "b", "b");
sym!(C_GLIDE, "c", "c");
sym!(D_GLIDE, "d", "d");
sym!(E_GLIDE, "e", "e");
sym!(N_GLIDE, "n", "n");
sym!(MIRROR, "m", "m");

sym!(SUB_1, "1", "\u{2081}");
sym!(SUB_2, "2", "\u{2082}");
sym!(SUB_3, "3", "\u{2083}");
sym!(SUB_4, "4", "\u{2084}");
sym!(SUB_5, "5", "\u{2085}");
sym!(SUB_6, "6", "\u{2086}");
sym!(SUB_7, "7", "\u{2087}");
sym!(SUB_8, "8", "\u{2088}");
sym!(SUB_9, "9", "\u{2089}");
sym!(SUB_0, "0", "\u{2080}");
sym!(SUB_PLUS, "+", "\u{208A}");
sym!(SUB_MINUS, "-", "\u{208B}");
sym!(SUB_EQUALS, "=", "\u{208C}");
sym!(SUB_LPAREN, "(", "\u{208D}");
sym!(SUB_RPAREN, ")", "\u{208E}");

sym!(SUP_1, "1", "\u{00B9}");
sym!(SUP_2, "2", "\u{00B2}");
sym!(SUP_3, "3", "\u{00B3}");
sym!(SUP_4, "4", "\u{2074}");
sym!(SUP_5, "5", "\u{2075}");
sym!(SUP_6, "6", "\u{2076}");
sym!(SUP_7, "7", "\u{2077}");
sym!(SUP_8, "8", "\u{2078}");
sym!(SUP_9, "9", "\u{2079}");
sym!(SUP_0, "0", "\u{2070}");
sym!(SUP_PLUS, "+", "\u{207A}");
sym!(SUP_MINUS, "-", "\u{207B}");
sym!(SUP_EQUALS, "=", "\u{207C}");
sym!(SUP_LPAREN, "(", "\u{207D}");
sym!(SUP_RPAREN, ")", "\u{207E}");

// The rules for superscripting and subscripting numbers are a total disaster in Unicode due to
// reasons. I hope the compiler knows how to make this fast.

/// Returns the Unicode subscript for a digit, if it exists, otherwise `None`.
pub const fn sub_digit(c: char) -> Option<Block> {
    match c {
        '0' => Some(SUB_0),
        '1' => Some(SUB_1),
        '2' => Some(SUB_2),
        '3' => Some(SUB_3),
        '4' => Some(SUB_4),
        '5' => Some(SUB_5),
        '6' => Some(SUB_6),
        '7' => Some(SUB_7),
        '8' => Some(SUB_8),
        '9' => Some(SUB_9),
        '+' => Some(SUB_PLUS),
        '-' => Some(SUB_MINUS),
        '=' => Some(SUB_EQUALS),
        '(' => Some(SUB_LPAREN),
        ')' => Some(SUB_RPAREN),
        _ => None,
    }
}

/// Returns the Unicode superscript for a digit, if it exists, otherwise `None`.
pub const fn super_digit(c: char) -> Option<Block> {
    match c {
        '0' => Some(SUP_0),
        '1' => Some(SUP_1),
        '2' => Some(SUP_2),
        '3' => Some(SUP_3),
        '4' => Some(SUP_4),
        '5' => Some(SUP_5),
        '6' => Some(SUP_6),
        '7' => Some(SUP_7),
        '8' => Some(SUP_8),
        '9' => Some(SUP_9),
        '+' => Some(SUP_PLUS),
        '-' => Some(SUP_MINUS),
        '=' => Some(SUP_EQUALS),
        '(' => Some(SUP_LPAREN),
        ')' => Some(SUP_RPAREN),
        _ => None,
    }
}
