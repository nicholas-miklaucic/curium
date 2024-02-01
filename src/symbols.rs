//! Symbols for Curium markup. Allow for use of Unicode when desired with ASCII fallback.

use crate::markup::Block;

macro_rules! sym {
    ($i:ident, $asc:literal, $uni:literal) => {
        pub const $i: Block = Block::new_symbol($asc, $uni);
    };
}

sym!(FRAC_SLASH, "/", "\u{2044}");
sym!(MINUS_SIGN, "-", "－");
sym!(PLUS_SIGN, "+", "＋");
sym!(X, "x", "\u{1D465}");
sym!(Y, "y", "\u{1D466}");
sym!(Z, "z", "\u{1D467}");
sym!(LANGLE, "<", "⟨");
sym!(RANGLE, ">", "⟩");
