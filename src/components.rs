//! Defines markup primitives for rendering output.

use either::Either;

use crate::markup::{Block, Render, RenderCanvas, RenderComponents, RenderMode};

/// A nonnegative integer.
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Uint(u128);

impl RenderComponents for Uint {
    type Components = Block;

    fn components(&self) -> Self::Components {
        Block::Text(format!("{}", self.0))
    }
}

impl<M: RenderMode, L: Render<M>, R: Render<M>> Render<M> for Either<L, R> {
    fn render_to<'a, 'b, C: RenderCanvas<M>>(&'a self, c: &'b mut C) -> &'b mut C {
        match self {
            Self::Left(l) => l.render_to(c),
            Self::Right(r) => r.render_to(c),
        }
    }
}

/// Signed integer.
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Int(i128);

impl RenderComponents for Int {
    type Components = Either<(Block, Uint), Uint>;

    fn components(&self) -> Self::Components {
        if self.0 < 0 {
            Either::Left((Block::MINUS_SIGN, Uint(self.0.unsigned_abs())))
        } else {
            Either::Right(Uint(self.0.unsigned_abs()))
        }
    }
}

/// The ITA style, adapted for terminals.
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ItaTerminal {}

impl RenderMode for ItaTerminal {}

#[cfg(test)]
mod tests {
    use crate::markup::Unicode;

    use super::*;

    #[test]
    fn test_negate() {
        assert_eq!(Render::<Unicode>::render_as_str(&Int(-5)).as_str(), "-5");

        assert_eq!(
            Render::<ItaTerminal>::render_as_str(&Int(-5)).as_str(),
            "5\u{0305}"
        );
    }
}
