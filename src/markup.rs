//! Functionality for displaying different Curium types to humans. The ITA book and crystallography
//! literature uses a concise format that doesn't easily translate to ASCII. It can be mocked in
//! Unicode, but that is brittle for some user environments. When preparing for publication, output
//! in LaTeX or Typst is instead required. Because of this, there's not just one way to turn Curium
//! types into a human-readable string. The solution is a unified markup format that allows Curium
//! types to format themselves in a single way and, from that, generate alternative presentations.
//! The resulting markup is also just useful for a variety of UI applications, even outside of the
//! specific needs of the library itself.

/// A primitive in Curium's markup system. Any type that can represent itself using these pieces
/// can easily be faithfully rendered in different formats.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// Plain text. Escaping is handled by individual modes: this should be formatted for humans.
    Text(String),
    /// An integer.
    Integer(i64),
    /// A floating-point number.
    Float(f64),
    /// A symbol that can be represented using Unicode or ASCII, such as an arrow →.
    Symbol(Symbol),
}

/// A symbol that can be represented using Unicode or ASCII, such as an arrow →.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Symbol {
    pub ascii: &'static str,
    pub unicode: &'static str,
}

impl Symbol {
    /// The fraction slash, which
    pub const FRAC_SLASH: Symbol = Symbol {
        ascii: "/",
        unicode: "⁄",
    };
}

impl Block {
    pub fn new_text<T: Into<String>>(string: T) -> Self {
        Block::Text(string.into())
    }

    pub fn new_int<T: Into<i64>>(int: T) -> Self {
        Block::Integer(int.into())
    }

    pub fn new_float<T: Into<f64>>(float: T) -> Self {
        Block::Float(float.into())
    }
}

/// A render *mode*, an environment that can display information.
pub trait RenderMode: Sized {
    /// Render a primitive.
    fn render_block(&mut self, block: &Block) -> String;
    /// Render a collection of primitives.
    fn render_blocks(&mut self, blocks: &[Block]) -> String {
        let outputs: Vec<String> = blocks.iter().map(|b| self.render_block(b)).collect();

        outputs.join("")
    }
}

pub trait BlockSequence {
    fn blocks(&self) -> Vec<Block>;
}

/// An element that can be rendered inside a rendering environment, or mode.
pub trait Render<M> {
    fn render(&self, mode: &mut M) -> String;
}

impl<M: RenderMode> Render<M> for Block {
    fn render(&self, mode: &mut M) -> String {
        mode.render_block(self)
    }
}

impl<V: BlockSequence, M: RenderMode> Render<M> for V {
    default fn render(&self, mode: &mut M) -> String {
        mode.render_blocks(&self.blocks())
    }
}

pub struct Ascii {}

impl RenderMode for Ascii {
    fn render_block(&mut self, block: &Block) -> String {
        match &block {
            Block::Text(t) => t.clone(),
            Block::Integer(i) => format!("{i}"),
            Block::Float(f) => format!("{f:.06}"),
            Block::Symbol(Symbol { ascii, unicode: _ }) => (*ascii).into(),
        }
    }
}

/// Render mode that attempts to mimic ITA style using non-ASCII characters.
#[derive(Debug, Default, Copy, Clone, Eq, Hash, PartialEq)]
pub struct ItaStyle {}

impl RenderMode for ItaStyle {
    fn render_block(&mut self, block: &Block) -> String {
        match block {
            Block::Text(t) => t.clone(),
            Block::Integer(i) => {
                if i < &0 && i >= &-9 {
                    // only use overline for single-digit negatives
                    let i_abs = i.abs();
                    format!("{i_abs}\u{305}")
                } else {
                    format!("{i}")
                }
            }
            Block::Float(f) => format!("{f:.06}"),
            Block::Symbol(Symbol { ascii: _, unicode }) => (*unicode).into(),
        }
    }
}

pub struct Arrow {}

impl BlockSequence for Arrow {
    fn blocks(&self) -> Vec<Block> {
        vec![Block::Text("⇒".into())]
    }
}

impl Render<Ascii> for Arrow {
    fn render(&self, mode: &mut Ascii) -> String {
        mode.render_block(&Block::Text("=>".into()))
    }
}

#[cfg(test)]
mod tests {
    use crate::frac;

    use super::*;

    #[test]
    fn test_specialized() {
        let a = Arrow {};
        let mut m: Ascii = Ascii {};

        assert_eq!(a.render(&mut m).as_str(), "=>");
    }

    #[test]
    fn test_frac() {
        let f = frac!(-1 / 4);
        let mut m = ItaStyle::default();
        println!("{} {}", f.render(&mut m), "\u{305}1\u{2044}4");
        assert_eq!(f.render(&mut m).as_str(), "\u{305}1\u{2044}6");
    }
}
