//! A Rational data type, developed specifically for fractional coordinates in the specification of
//! coordinate transformations and space groups.

use std::fmt::Display;

/// The base type used. We don't need large values here, ±32768 is more than enough.
type BaseInt = i16;
/// The base needed to represent all of the necessary coordinate transformations and symmetry
/// operations. 24 is used by GEMMI and should be all we need.
const DENOM: BaseInt = 24;

/// A fraction with a hardcoded denominator [`DENOM`]. Used to ensure numerical stability and
/// eliminate rounding errors.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Default)]
pub struct Frac {
    /// The numerator.
    numerator: BaseInt
}

impl Frac {
    fn new(numerator: BaseInt) -> Self {
        Self{numerator}
    }
}

// If this fails, then some of the special cases in the Display impl below will be incorrect.
assert!(DENOM % 24 == 0);

/// When printing out a fraction, we can make things more compact by only using / when necessary and
/// otherwise using the existing Unicode characters for fractions.
static UNICODE_SPECIAL_CASES: phf::Map<Frac, &'static str> = phf_map!{
    Frac::new{0} => "0",
    Frac::new{DENOM} => "1",
    Frac::new{DENOM / 2} => "\u{00BD}",
    Frac::new{DENOM / 3} => "\u{2153}",
    Frac::new{DENOM / 4} => "\u{00BC}",
    Frac::new{DENOM / 6} => "\u{2159}",
    Frac::new{DENOM / 8} => "\u{215B}",
    Frac::new{2 * DENOM / 3} => "\u{2154}",
    Frac::new{3 * DENOM / 4} => "\u{00BE}",
    Frac::new{3 * DENOM / 8} => "\u{215C}",
    Frac::new{5 * DENOM / 8} => "\u{215D}",
    Frac::new{7 * DENOM / 8} => "\u{215E}",
    Frac::new{5 * DENOM / 6} => "\u{215A}",
};

impl Display for Frac {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let abs_num = self.numerator.abs();
        let prefix = if self.numerator < 0 {
            "\u{0305}" // combining overline
        } else {
            ""
        };


        write!(f, "{}")
    }
}