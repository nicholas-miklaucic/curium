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

/// Signed integer.
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Int(i128);

impl RenderComponents for Int {
    type Components = Either<Negate<Uint>, Uint>;

    fn components(&self) -> Self::Components {
        if self.0 < 0 {
            Either::Left(Negate::new(Uint(self.0.unsigned_abs())))
        } else {
            Either::Right(Uint(self.0.unsigned_abs()))
        }
    }
}

/// Negation, usually represented with a minus sign but
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct Negate<T>(T);

impl<T: > Negate<T> {
    fn new(t: T) -> Self {
        Self { 0: t }
    }
}

/// The ITA style, adapted for terminals.
#[derive(Debug, Default, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ItaTerminal {}

impl RenderMode for ItaTerminal {}

impl<T: Render<ItaTerminal>> Render<ItaTerminal> for Negate<T> {
    fn render_to<'a, 'b, C: RenderCanvas<ItaTerminal>>(&'a self, c: &'b mut C) -> &'b mut C {
        let out = self.0.render_as_str();
        let neg_out = out.chars().flat_map(|c| [c, '\u{0305}']);
        let neg_out: String = neg_out.collect();
        c.render_raw(neg_out.as_str());
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negate() {

    }
}