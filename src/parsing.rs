//! Utilities for parsing expressions from a standard ASCII syntax that is easy to type.

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{i8, one_of};
use nom::combinator::{fail, map, opt, value};
use nom::error::{Error, ParseError};
use nom::multi::{many0, many1, separated_list0};
use nom::sequence::{separated_pair, tuple};
use nom::IResult;

use crate::frac;
use crate::fract::BaseInt;
use crate::hall::{
    HallAxisSymbol, HallCenteringType, HallGroupSymbol, HallOpSymbol, HallTranslationSymbol,
    RotationGroup,
};
use crate::lattice::CenteringType;
use crate::spacegroupdata::{FullHMSymbolUnit, PartialSymmOp, RotOrder};
use crate::symmop::{RotationDirectness, ScrewOrder};

use nom::{character::complete::multispace0, sequence::delimited};

/// Parses a complete Hall symbol for a space group.
pub fn hall_group(input: &str) -> IResult<&str, HallGroupSymbol> {
    let (o, (has_inv, center, gens, shift)) = tuple((
        ws(hall_sign),
        ws(hall_centering),
        many1(ws(hall_generator)),
        ws(hall_origin_shift),
    ))(input)?;

    let mut hall_gens: Vec<HallOpSymbol> = vec![];

    for (rot, ax, tau) in gens {
        // first axis defaults to c
        hall_gens.push(HallOpSymbol {
            rotation: rot,
            axis: ax,
            translation: tau,
        });
    }

    Ok((
        o,
        HallGroupSymbol {
            has_inversion: has_inv,
            centering: center,
            generators: hall_gens,
            shift: shift.unwrap_or((0, 0, 0)),
        },
    ))
}

/// Parses a complete Hall rotational operation.
pub fn hall_generator(
    input: &str,
) -> IResult<
    &str,
    (
        RotationGroup,
        Option<HallAxisSymbol>,
        Vec<HallTranslationSymbol>,
    ),
> {
    tuple((
        hall_rot_group,
        opt(hall_axis_symbol),
        many0(hall_translation_symbol),
    ))(input)
}

/// Parses a Hall origin shift.
pub fn hall_origin_shift(input: &str) -> IResult<&str, Option<(i8, i8, i8)>> {
    opt(delimited(
        tag("("),
        tuple((ws(i8), ws(i8), ws(i8))),
        tag(")"),
    ))(input)
}

/// Parses a Hall axis symbol.
pub fn hall_axis_symbol(input: &str) -> IResult<&str, HallAxisSymbol> {
    // TODO double prime and prime are flipped
    alt((
        value(HallAxisSymbol::X, one_of("Xx")),
        value(HallAxisSymbol::Y, one_of("Yy")),
        value(HallAxisSymbol::Z, one_of("Zz")),
        value(HallAxisSymbol::Prime, one_of("'")),
        value(HallAxisSymbol::DoublePrime, one_of("\"")),
        value(HallAxisSymbol::Star, one_of("*")),
    ))(input)
}

/// Parses a Hall translation symbol.
pub fn hall_translation_symbol(input: &str) -> IResult<&str, HallTranslationSymbol> {
    alt((
        value(HallTranslationSymbol::A, one_of("Aa")),
        value(HallTranslationSymbol::B, one_of("Bb")),
        value(HallTranslationSymbol::C, one_of("Cc")),
        value(HallTranslationSymbol::U, one_of("Uu")),
        value(HallTranslationSymbol::V, one_of("Vv")),
        value(HallTranslationSymbol::W, one_of("Ww")),
        value(HallTranslationSymbol::N, one_of("Nn")),
        value(HallTranslationSymbol::D, one_of("Dd")),
    ))(input)
}

/// Parses a Hall rotation group: an optional sign, rotation order, and screw order.
pub fn hall_rot_group(input: &str) -> IResult<&str, RotationGroup> {
    let (o, (sign, rot, screw)) = tuple((hall_sign, hall_rot, hall_screw))(input)?;

    let sign = if sign {
        RotationDirectness::Improper
    } else {
        RotationDirectness::Proper
    };

    let screw = frac!(screw) / frac!(rot);

    let group = RotationGroup::try_new(rot, sign, screw).unwrap();

    Ok((o, group))
}

/// Parses a Hall sign: either a minus sigh or nothing.
pub fn hall_sign(input: &str) -> IResult<&str, bool> {
    opt(minus_sign)(input).map(|(i, o)| (i, o.is_some()))
}

/// Parses a Hall rotation order, a number from 1 to 6.
pub fn hall_rot(input: &str) -> IResult<&str, RotOrder> {
    alt((
        value(1, tag("1")),
        value(2, tag("2")),
        value(3, tag("3")),
        value(4, tag("4")),
        value(5, tag("5")),
        value(6, tag("6")),
    ))(input)
}

/// Parses a Hall screw, with an underscore or without.
pub fn hall_screw(input: &str) -> IResult<&str, BaseInt> {
    let (o, (_underscore, s)) = tuple((
        opt(tag("_")),
        opt(alt((
            value(1, tag("1")),
            value(2, tag("2")),
            value(3, tag("3")),
            value(4, tag("4")),
            value(5, tag("5")),
            value(6, tag("6")),
        ))),
    ))(input)?;

    Ok((o, s.unwrap_or(0)))
}

/// Parses a Hall centering symbol, without a sign.
pub fn hall_centering(input: &str) -> IResult<&str, HallCenteringType> {
    alt((
        value(HallCenteringType::P, tag("P")),
        value(HallCenteringType::A, tag("A")),
        value(HallCenteringType::B, tag("B")),
        value(HallCenteringType::C, tag("C")),
        value(HallCenteringType::I, tag("I")),
        value(HallCenteringType::R, tag("R")),
        value(HallCenteringType::S, tag("S")),
        value(HallCenteringType::T, tag("T")),
        value(HallCenteringType::F, tag("F")),
    ))(input)
}

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

    Ok((o, PartialSymmOp::GenRotation(r * sign, frac!(0))))
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
        Ok((o, PartialSymmOp::GenRotation(r, frac!(s) / frac!(r))))
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
        value(PartialSymmOp::GenRotation(-2, frac!(0)), tag("m")),
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

    use crate::markup::ASCII;

    use super::*;

    #[test]
    fn test_hall_parse_roundtrip() {
        for symbol in [
            "-P 2av 2buc -1wnd (-1 0 2)",
            "P 31 2c (0 0 1)",
            "P 31 2",
            "P 3* 2",
            "-F 2uv 2vw 3",
            "-F 4cvw 2vw 3",
            "P 2 2 -1n",
            "-C 2xbc",
        ] {
            let (o, hall) = hall_group(symbol).unwrap();
            assert_eq!(o, "");
            assert_eq!(ASCII.render_to_string(&hall).as_str(), symbol);
        }
    }

    #[test]
    fn test_hall_parse() {
        let (o, hall) = hall_group("-P 2av 2buc -1wnd (-1 0 2)").unwrap();
        assert_eq!(o, "");
        assert_eq!(
            hall,
            HallGroupSymbol {
                has_inversion: true,
                centering: HallCenteringType::P,
                generators: vec![
                    HallOpSymbol {
                        rotation: RotationGroup::R2,
                        axis: None,
                        translation: vec![HallTranslationSymbol::A, HallTranslationSymbol::V],
                    },
                    HallOpSymbol {
                        rotation: RotationGroup::R2,
                        axis: None,
                        translation: vec![
                            HallTranslationSymbol::B,
                            HallTranslationSymbol::U,
                            HallTranslationSymbol::C
                        ],
                    },
                    HallOpSymbol {
                        rotation: RotationGroup::Rm1,
                        axis: None,
                        translation: vec![
                            HallTranslationSymbol::W,
                            HallTranslationSymbol::N,
                            HallTranslationSymbol::D
                        ],
                    },
                ],
                shift: (-1, 0, 2,),
            }
        );
    }

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
