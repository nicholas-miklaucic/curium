//! Functionality for displaying different Curium types to humans. The ITA book and crystallography
//! literature uses a concise format that doesn't easily translate to ASCII. It can be mocked in
//! Unicode, but that is brittle for some user environments. When preparing for publication, output
//! in LaTeX or Typst is instead required. Because of this, there's not just one way to turn Curium
//! types into a human-readable string. The solution is a unified markup format that allows Curium
//! types to format themselves in a single way and, from that, generate alternative presentations.
//! The resulting markup is also just useful for a variety of UI applications, even outside of the
//! specific needs of the library itself.

use crate::frac::Frac;
use fortuples::fortuples;

/// A primitive in Curium's markup system. Any type that can represent itself using these pieces
/// can easily be faithfully rendered in different formats.
#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    /// Plain text. Escaping is handled by individual modes: this should be formatted for humans.
    Text(String),
    /// A symbol with multiple representations depending on available characters and environment,
    /// such as a right arrow.
    Symbol(Symbol),
    /// A concatenation of blocks that are inseparable, like two symbols one after another.
    Concatenation(Vec<Block>),
}

/// A symbol that can be represented using Unicode or ASCII, such as an arrow →.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Symbol {
    pub ascii: &'static str,
    pub unicode: &'static str,
}

impl Symbol {}

impl Block {
    pub fn new_text<T: Into<String>>(string: T) -> Self {
        Block::Text(string.into())
    }

    pub const fn new_symbol(ascii: &'static str, unicode: &'static str) -> Self {
        Block::Symbol(Symbol { ascii, unicode })
    }

    pub fn new_concatenation<T: IntoIterator<Item = Block>>(t: T) -> Self {
        Block::Concatenation(t.into_iter().collect())
    }

    pub const NONE: Block = Block::Text(String::new());

    /// The fraction slash, represented using a normal slash in ASCII mode.
    pub const FRAC_SLASH: Block = Block::new_symbol("/", "\u{2044}");

    /// The minus sign, falling back to a hyphen.
    pub const MINUS_SIGN: Block = Block::new_symbol("-", "\u{2212}");
}

/// A render *mode*: an environment in which information can be displayed.
pub trait RenderMode: Default {
    /// Renders plain text. This is where, for instance, escaping should happen: it's assumed that
    /// any characters in the input are intended to be represented as they are, not as control
    /// codes or markup.
    fn render_text(&mut self, text: &str) -> String {
        text.to_string()
    }

    /// Renders a symbol.
    fn render_symbol(&mut self, sym: &Symbol) -> String {
        sym.unicode.to_string()
    }

    /// Renders a concatenation of other blocks. The default should work (one after another) most of
    /// the time, but some modes may want to add spaces (if those aren't semantic) or do other
    /// escaping.
    fn render_concatenation<'a, C: IntoIterator<Item = &'a Block>>(&mut self, blocks: C) -> String {
        blocks.into_iter().map(|b| self.render_block(b)).collect()
    }

    /// Render a block.
    ///
    /// When implementing [`RenderMode`], generally do not override this method. Instead, implement
    /// whichever branch methods you want to customize.
    fn render_block(&mut self, block: &Block) -> String {
        match block {
            Block::Text(t) => self.render_text(t),
            Block::Symbol(sym) => self.render_symbol(sym),
            Block::Concatenation(blocks) => self.render_concatenation(blocks),
        }
    }
}

/// Trait representing the state of a document in progress that can be written to iteratively and
/// then completed, returning a `String`.
pub trait RenderCanvas<M: RenderMode> {
    /// Initializes an empty document with the given mode.
    fn new_empty() -> Self;

    /// Renders an element.
    fn render_block(&mut self, block: &Block) -> &mut Self;

    /// Pushes a raw string to the buffer. Use with caution!
    fn render_raw(&mut self, string: &str) -> &mut Self;

    /// Completes writing, returning the output String.
    fn complete(self) -> String;
}

/// A buffer into which elements can be rendered. Essentially a [`RenderMode`] that can change,
/// backed by a string. Does not keep track of rendered state outside the mode.
#[derive(Debug, Default, Clone)]
pub struct SimpleStringBuf<M: RenderMode> {
    buf: String,
    mode: M,
}

impl<M: RenderMode> RenderCanvas<M> for SimpleStringBuf<M> {
    fn new_empty() -> Self {
        Self {
            buf: String::new(),
            mode: M::default(),
        }
    }

    /// Renders an element.
    fn render_block(&mut self, block: &Block) -> &mut Self {
        self.buf.push_str(self.mode.render_block(block).as_str());
        self
    }

    /// Pushes a raw string to the buffer. Use with caution!
    fn render_raw(&mut self, string: &str) -> &mut Self {
        self.buf.push_str(string);
        self
    }

    /// Completes writing, returning the output String.
    fn complete(self) -> String {
        self.buf
    }
}

pub trait Render<M: RenderMode> {
    fn render_to<'a, 'b, C: RenderCanvas<M>>(&'a self, c: &'b mut C) -> &'b mut C;

    /// Renders to an empty string.
    fn render_as_str(&self) -> String {
        let mut buf = SimpleStringBuf::<M>::new_empty();
        self.render_to(&mut buf);
        buf.complete()
    }
}

impl<M> Render<M> for Block
where
    M: RenderMode,
{
    fn render_to<'a, 'b, C: RenderCanvas<M>>(&'a self, c: &'b mut C) -> &'b mut C {
        c.render_block(self)
    }
}

/// Trait for types that can be rendered by concatenating smaller pieces.
pub trait RenderComponents {
    /// The type of the components.
    type Components;

    /// Decompose self into components.
    fn components(&self) -> Self::Components;
}

fortuples! {
    #[tuples::min_size(1)]
    impl<M: RenderMode> Render<M> for #Tuple
    where
        #(#Member: Render<M>),*
    {
        fn render_to<'a, 'b, C: RenderCanvas<M>>(&'a self, c: &'b mut C) -> &'b mut C {
            #(#self.render_to(c);)*

            c
        }
    }
}

impl<T, M> Render<M> for T
where
    T: RenderComponents,
    T::Components: Render<M>,
    M: RenderMode,
{
    default fn render_to<'a, 'b, C: RenderCanvas<M>>(&'a self, c: &'b mut C) -> &'b mut C {
        self.components().render_to(c)
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Typst {}

impl RenderMode for Typst {}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Unicode {}

impl RenderMode for Unicode {}

impl RenderComponents for Frac {
    type Components = (Block, Block, Block);

    default fn components(&self) -> Self::Components {
        (
            Block::Text(format!("{}", self.numerator)),
            Block::FRAC_SLASH,
            Block::Text(format!("{}", Self::DENOM)),
        )
    }
}

impl Render<Typst> for Frac {
    fn render_to<'a, 'b, C: RenderCanvas<Typst>>(&'a self, c: &'b mut C) -> &'b mut C {
        c.render_raw(format!("({})/({})", self.numerator, Self::DENOM).as_str())
    }
}

#[cfg(test)]
mod tests {
    use crate::frac;

    use super::*;

    #[test]
    fn test_frac() {
        assert_eq!(
            Render::<Unicode>::render_as_str(&frac!(5 / 24)).as_str(),
            "5\u{2044}24"
        );

        assert_eq!(
            Render::<Typst>::render_as_str(&frac!(5 / 24)).as_str(),
            "(5)/(24)"
        );
    }
}
