//! Functionality for displaying different Curium types to humans. The ITA book and crystallography
//! literature uses a concise format that doesn't easily translate to ASCII. It can be mocked in
//! Unicode, but that is brittle for some user environments. When preparing for publication, output
//! in LaTeX or Typst is instead required. Because of this, there's not just one way to turn Curium
//! types into a human-readable string. The solution is a unified markup format that allows Curium
//! types to format themselves in a single way and, from that, generate alternative presentations.
//! The resulting markup is also just useful for a variety of UI applications, even outside of the
//! specific needs of the library itself.

use crate::symbols::*;
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
    /// An unsigned integer. Should be used with `Signed` unless signs are never needed.
    Uint(u64),
    /// A unsigned float. Should be used with `Signed` unless signs are never needed.
    Ufloat(f64),
    /// A signed number. Assumes the first input is the *unsigned* version.
    Signed(Box<Block>, Sign),
    /// A fraction with numerator and denominator.
    Fraction(Box<Block>, Box<Block>),
    /// A point in 3D space.
    Point3D(Box<Block>, Box<Block>, Box<Block>),
    /// A mathematical vector.
    Vector(Vec<Block>),
    /// A collection of blocks.
    Blocks(Vec<Block>),
}

/// The sign of a number. This is typographic, not numeric: -0 is not the same as +0.
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
pub enum Sign {
    /// Positive, with either `+` or nothing.
    Positive,
    /// Negative, with `-`.
    Negative,
}

/// A symbol that can be represented using Unicode or ASCII, such as an arrow →.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct Symbol {
    pub ascii: &'static str,
    pub unicode: &'static str,
}

impl Symbol {}

impl Block {
    pub fn new_text(text: &str) -> Self {
        Self::Text(text.to_string())
    }

    pub const fn new_symbol(ascii: &'static str, unicode: &'static str) -> Self {
        Self::Symbol(Symbol { ascii, unicode })
    }

    pub fn new_uint(uint: u64) -> Self {
        Self::Uint(uint)
    }

    pub fn new_int(int: i64) -> Self {
        let uint = Self::new_uint(int.unsigned_abs());
        if int < 0 {
            Self::Signed(uint.into(), Sign::Negative)
        } else {
            Self::Signed(uint.into(), Sign::Positive)
        }
    }

    pub fn new_float(float: f64) -> Self {
        let ufloat = Self::Ufloat(float.abs());
        if float < 0. {
            Self::Signed(ufloat.into(), Sign::Negative)
        } else {
            Self::Signed(ufloat.into(), Sign::Positive)
        }
    }
}

/// A document in which information can be rendered.
pub trait RenderDoc {
    /// Writes raw text to the document. Used to support simple overriding of the storage for text
    /// without changing the way information is serialized.
    fn write_raw<T: AsRef<str>>(&mut self, raw: T) -> &mut Self;

    /// Returns the final document as a `String`. This is where preambles, linking, TOCs, or similar
    /// document-level processing should be done.
    fn complete(self) -> String;

    /// Renders plain text. This is where, for instance, escaping should happen: it's assumed that
    /// any characters in the input are intended to be represented as they are, not as control
    /// codes or markup.
    fn render_text(&mut self, text: &str) -> &mut Self {
        self.write_raw(text)
    }

    /// Renders a symbol.
    fn render_symbol(&mut self, sym: &Symbol) -> &mut Self {
        self.write_raw(sym.unicode)
    }

    /// Renders an unsigned integer.
    fn render_uint(&mut self, u: u64) -> &mut Self {
        self.write_raw(u.to_string())
    }

    /// Renders an unsigned float.
    fn render_ufloat(&mut self, f: f64) -> &mut Self {
        self.write_raw(f.to_string())
    }

    /// Renders a signed number.
    fn render_signed(&mut self, block: &Block, sign: &Sign) -> &mut Self {
        match sign {
            Sign::Positive => self.render_block(block),
            Sign::Negative => self.render_block(&MINUS_SIGN).render_block(block),
        }
    }

    /// Renders a fraction.
    fn render_fraction(&mut self, num: &Block, denom: &Block) -> &mut Self {
        self.render_block(num)
            .render_block(&FRAC_SLASH)
            .render_block(denom)
    }

    /// Renders a point in 3D space.
    fn render_point3d(&mut self, x: &Block, y: &Block, z: &Block) -> &mut Self {
        self.render_blocks(&[
            Block::new_text("("),
            x.clone(),
            Block::new_text(","),
            y.clone(),
            Block::new_text(","),
            z.clone(),
            Block::new_text(")"),
        ])
    }

    /// Renders a vector.
    fn render_vector(&mut self, blocks: &[Block]) -> &mut Self {
        let mut elements = vec![];
        elements.push(Block::new_text("<"));
        for block in blocks {
            elements.push(block.clone());
            elements.push(Block::new_text(","));
        }
        elements.pop();
        elements.push(Block::new_text(">"));
        self.render_blocks(&elements)
    }

    /// Render a block.
    ///
    /// When implementing [`RenderDoc`], generally do not override this method.
    /// Instead, implement whichever branch methods you want to customize.
    fn render_block(&mut self, block: &Block) -> &mut Self {
        match block {
            Block::Text(text) => self.render_text(text),
            Block::Symbol(sym) => self.render_symbol(sym),
            Block::Uint(u) => self.render_uint(*u),
            Block::Ufloat(f) => self.render_ufloat(*f),
            Block::Signed(block, sign) => self.render_signed(block, sign),
            Block::Fraction(num, denom) => self.render_fraction(num, denom),
            Block::Point3D(x, y, z) => self.render_point3d(x, y, z),
            Block::Blocks(blocks) => self.render_blocks(blocks),
            Block::Vector(blocks) => self.render_vector(blocks),
        }
    }

    /// Renders an iterator of blocks, one after another.
    fn render_blocks<'a, I>(&mut self, blocks: I) -> &mut Self
    where
        I: IntoIterator<Item = &'a Block>,
    {
        for block in blocks {
            self.render_block(block);
        }
        self
    }

    /// Renders any element.
    fn render_element<T: Render>(&mut self, el: &T) -> &mut Self
    where
        Self: Sized,
    {
        el.render_to(self)
    }
}

/// A render environment specification.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct SimpleRenderConfig {
    /// Whether Unicode characters are supported. If `false`, only ASCII is used wherever possible.
    pub unicode: bool,
    /// Whether the ITA combining overline notation is used to negate.
    pub ita_minus: bool,
}

impl SimpleRenderConfig {
    pub fn render_to_string<T: Render>(&self, el: &T) -> String {
        let mut doc = SimpleRenderDoc::new_empty(self.clone());
        doc.render_element(el);
        doc.complete()
    }
}

/// A base implementation of [`RenderDoc`] that allows for basic configuration. Writes to a
/// [`String`].
pub struct SimpleRenderDoc {
    /// The buffer.
    buf: String,
    /// The feature flags for the renderer.
    config: SimpleRenderConfig,
}

impl SimpleRenderDoc {
    pub fn new_empty(conf: SimpleRenderConfig) -> Self {
        Self {
            buf: String::new(),
            config: conf,
        }
    }
}

impl RenderDoc for SimpleRenderDoc {
    fn write_raw<T: AsRef<str>>(&mut self, raw: T) -> &mut Self {
        self.buf.push_str(raw.as_ref());
        self
    }

    fn complete(self) -> String {
        self.buf
    }

    fn render_text(&mut self, text: &str) -> &mut Self {
        self.write_raw(text)
    }

    fn render_symbol(&mut self, sym: &Symbol) -> &mut Self {
        if self.config.unicode {
            self.write_raw(sym.unicode)
        } else {
            self.write_raw(sym.ascii)
        }
    }

    fn render_uint(&mut self, u: u64) -> &mut Self {
        self.write_raw(u.to_string())
    }

    fn render_ufloat(&mut self, f: f64) -> &mut Self {
        self.write_raw(f.to_string())
    }

    fn render_signed(&mut self, block: &Block, sign: &Sign) -> &mut Self {
        if self.config.ita_minus && self.config.unicode {
            let join_char = match sign {
                Sign::Positive => '\u{200b}',
                // zero-width space: trick nalgebra into formatting correctly
                Sign::Negative => '\u{0305}',
            };
            let abs = self.config.render_to_string(block);
            self.write_raw(abs.chars().flat_map(|c| [join_char, c]).collect::<String>())
        } else if let Sign::Positive = sign {
            self.render_block(block)
        } else {
            self.render_blocks(&[MINUS_SIGN, block.clone()])
        }
    }

    fn render_fraction(&mut self, num: &Block, denom: &Block) -> &mut Self {
        self.render_block(num)
            .render_block(&FRAC_SLASH)
            .render_block(denom)
    }
}

/// Trait for objects renderable in Curium's markup as primitives.
pub trait RenderBlocks {
    fn components(&self) -> Vec<Block>;
}

impl RenderBlocks for Block {
    fn components(&self) -> Vec<Block> {
        vec![self.clone()]
    }
}

impl RenderBlocks for Vec<Block> {
    fn components(&self) -> Vec<Block> {
        self.clone()
    }
}

impl<const N: usize> RenderBlocks for &[Block; N] {
    fn components(&self) -> Vec<Block> {
        self.to_vec()
    }
}

macro_rules! render_uints {
    ($t:ident) => {
        impl RenderBlocks for $t {
            fn components(&self) -> Vec<Block> {
                vec![Block::new_uint(*self as u64)]
            }
        }
    };
}
render_uints!(u8);
render_uints!(u16);
render_uints!(u32);
render_uints!(u64);
render_uints!(u128);

macro_rules! render_ints {
    ($t:ident) => {
        impl RenderBlocks for $t {
            fn components(&self) -> Vec<Block> {
                vec![Block::new_int(*self as i64)]
            }
        }
    };
}
render_ints!(i8);
render_ints!(i16);
render_ints!(i32);
render_ints!(i64);

macro_rules! render_floats {
    ($t:ident) => {
        impl RenderBlocks for $t {
            fn components(&self) -> Vec<Block> {
                vec![Block::new_float(*self as f64)]
            }
        }
    };
}
render_floats!(f32);
render_floats!(f64);

fortuples! {
    #[tuples::min_size(1)]
    impl RenderBlocks for #Tuple
    where #(#Member: RenderBlocks),* {
        fn components(&self) -> Vec<Block> {
            let mut blocks = vec![];
            #(blocks.extend_from_slice(&#self.components());)*
            blocks
        }
    }
}

/// Trait for objects that are rendered in any fashion. Can support more complex logic than simple
/// decomposition, but in general this should not be implemented directly.
pub trait Render {
    fn render_to<'b, D: RenderDoc>(&self, d: &'b mut D) -> &'b mut D;
}

impl<T: RenderBlocks> Render for T {
    fn render_to<'b, D: RenderDoc>(&self, d: &'b mut D) -> &'b mut D {
        d.render_blocks(&self.components())
    }
}

pub const ASCII: SimpleRenderConfig = SimpleRenderConfig {
    unicode: false,
    ita_minus: false,
};

pub const UNICODE: SimpleRenderConfig = SimpleRenderConfig {
    unicode: true,
    ita_minus: false,
};

pub const ITA: SimpleRenderConfig = SimpleRenderConfig {
    unicode: true,
    ita_minus: true,
};

pub const DISPLAY: SimpleRenderConfig = ITA;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negate() {
        assert_eq!(UNICODE.render_to_string(&-123).as_str(), "\u{2212}123");

        assert_eq!(ITA.render_to_string(&-123).as_str(), "1̅2̅3̅");
    }

    #[test]
    fn test_tuple() {
        assert_eq!(UNICODE.render_to_string(&(1, 2)).as_str(), "12");
    }
}
