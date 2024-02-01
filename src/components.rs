//! Defines markup primitives for rendering output.

use crate::{
    frac::Frac,
    markup::{Block, Render, RenderCanvas, RenderComponents, RenderMode, Signed, Uint},
};

/// Signed integer.
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Int(pub i128);

impl RenderComponents for Int {
    type Components = Signed<Uint>;

    fn components(&self) -> Self::Components {
        let unsigned = Uint(self.0.unsigned_abs());
        Signed::new(unsigned, self.0 < 0)
    }
}

impl RenderComponents for Frac {
    type Components = Signed<(Uint, Block, Uint)>;

    fn components(&self) -> Self::Components {
        Signed::new(
            (
                Uint(self.numerator.unsigned_abs() as u128),
                Block::FRAC_SLASH,
                Uint(Self::DENOM.unsigned_abs() as u128),
            ),
            self.numerator < 0,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        frac,
        markup::{ItaTerminal, Unicode},
    };

    use super::*;

    #[test]
    fn test_frac() {
        assert_eq!(
            Render::<Unicode>::render_as_str(&frac!(5 / 24)).as_str(),
            "5\u{2044}24"
        );

        assert_eq!(
            Render::<ItaTerminal>::render_as_str(&frac!(5 / 24)).as_str(),
            "(5)/(24)"
        );
    }
}
