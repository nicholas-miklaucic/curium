//! Utilities for parsing expressions from a standard ASCII syntax that is easy to type.

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::one_of;
use nom::combinator::{fail, map, opt, value};
use nom::error::{Error, ParseError};
use nom::multi::{many1, separated_list0};
use nom::sequence::{separated_pair, tuple};
use nom::IResult;

use crate::lattice::CenteringType;
use crate::spacegroupdata::{FullHMSymbolUnit, PartialSymmOp};

use nom::{character::complete::multispace0, sequence::delimited};

/// Parses a short or long HM symbol into the centering type and units.
pub fn hm_symbol(input: &str) -> IResult<&str, (CenteringType, Vec<FullHMSymbolUnit>)> {
    tuple((ws(hm_centering_type), many1(ws(hm_unit))))(input)
}

/// Parses either a single or double HM operation.
pub fn hm_unit(input: &str) -> IResult<&str, FullHMSymbolUnit> {
    alt((
        double_hm_op,
        map(hm_partial_op, |p| {
            let mut f = FullHMSymbolUnit::default();
            f.and(p);
            f
        }),
    ))(input)
}

/// Parses a double HM operation separated by a slash.
pub fn double_hm_op(input: &str) -> IResult<&str, FullHMSymbolUnit> {
    let (o, (op1, op2)) = separated_pair(alt((hm_screw, hm_rot)), ws(slash), hm_reflection)(input)?;

    if !(!op1.is_reflection() && op2.is_reflection()) {
        fail(input)
    } else {
        let mut full = FullHMSymbolUnit::default();
        full.and(op1);
        full.and(op2);

        Ok((o, full))
    }
}

/// Parses a single partial symmetry operation, one component of a unit of an HM symbol. Accepts
/// underscores for screws, but does not require them.
pub fn hm_partial_op(input: &str) -> IResult<&str, PartialSymmOp> {
    alt((hm_screw, hm_reflection, hm_rot))(input)
}

/// A combinator that takes a parser `inner` and produces a parser that also consumes both leading and
/// trailing whitespace, returning the output of `inner`.
fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(multispace0, inner, multispace0)
}

/// Parses a centering type, the first letter of a Hermann-Mauguin symbol.
pub fn hm_centering_type(input: &str) -> IResult<&str, CenteringType> {
    alt((
        value(CenteringType::Primitive, tag("P")),
        value(CenteringType::ACentered, tag("A")),
        value(CenteringType::BCentered, tag("B")),
        value(CenteringType::CCentered, tag("C")),
        value(CenteringType::BodyCentered, tag("I")),
        value(CenteringType::FaceCentered, tag("F")),
        value(CenteringType::Rhombohedral, tag("R")),
    ))(input)
}

/// Parses a simple rotation without a screw, like 2 or -4.
pub fn hm_rot(input: &str) -> IResult<&str, PartialSymmOp> {
    let (o, (m, r)) = tuple((opt(one_of("-﹣－−")), one_of("12346")))(input)?;

    let sign = if m.is_some() { -1 } else { 1 };

    let r: i8 = r.to_string().parse().unwrap();

    Ok((o, PartialSymmOp::GenRotation(r * sign, 0)))
}

/// Parses a screw, assuming that there is never a screw rotoinversion. Can accept with or without
/// an underscore.
pub fn hm_screw(input: &str) -> IResult<&str, PartialSymmOp> {
    let (o, (r, _underscore, s)) =
        tuple((one_of("2346"), opt(one_of("_")), one_of("12345")))(input)?;

    let r: i8 = r.to_string().parse().unwrap();
    let s: i8 = s.to_string().parse().unwrap();

    if !(1..r).contains(&s) {
        fail(input)
    } else {
        Ok((o, PartialSymmOp::GenRotation(r, s)))
    }
}

/// Parses a glide or mirror in an HM symbol.
pub fn hm_reflection(input: &str) -> IResult<&str, PartialSymmOp> {
    alt((
        value(PartialSymmOp::AGlide, tag("a")),
        value(PartialSymmOp::BGlide, tag("b")),
        value(PartialSymmOp::CGlide, tag("c")),
        value(PartialSymmOp::DGlide, tag("d")),
        value(PartialSymmOp::EGlide, tag("e")),
        value(PartialSymmOp::NGlide, tag("n")),
        value(PartialSymmOp::GenRotation(-2, 0), tag("m")),
    ))(input)
}

/// Parses a minus sign, in many formats.
pub fn minus_sign(input: &str) -> IResult<&str, char> {
    one_of("-﹣－−")(input)
}

/// Parses a slash, as of a fraction.
pub fn slash(input: &str) -> IResult<&str, char> {
    one_of("/⁄∕")(input)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_hm_parse_pbcm() {
        let (o, (ctype, units)) = hm_symbol("P b c m").unwrap();
        assert_eq!(o, "");
        assert_eq!(ctype, CenteringType::Primitive);
        assert_eq!(
            &units,
            &[
                FullHMSymbolUnit::B,
                FullHMSymbolUnit::C,
                FullHMSymbolUnit::M,
            ]
        );
    }
}
