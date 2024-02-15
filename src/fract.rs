//! A Rational data type, developed specifically for fractional coordinates in the specification of
//! coordinate transformations and space groups.

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{ComplexField, Field, RealField, SimdValue};
use num_traits::{Float, FromPrimitive, Num, One, PrimInt, Signed, Zero};
use simba::{scalar::SubsetOf, simd::PrimitiveSimdValue};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
    str::FromStr,
};
use thiserror::Error;

use crate::markup::{Block, RenderBlocks, DISPLAY};

/// The base type used. We don't need large values here, ±32768 is more than enough.
pub type BaseInt = i16;
/// The base needed to represent all of the necessary coordinate transformations and symmetry
/// operations. 24 is used by GEMMI and should be all we need.

// The Display code and tests need to change if this changes.
pub const DENOM: BaseInt = 24;

/// The tolerance used to convert floats to `Frac`s.

// This needs to be high enough to allow generous numerical errors from e.g., linear algebra
// operations. It needs to be low enough to catch a deadly kind of error, if Frac is being used
// improperly and the resulting float is not actually a fraction with DENOM. (This needs to be
// significantly smaller than 0.5/DENOM, for instance, because otherwise all floats would round to
// the nearest value even though 3/48 is not representable as a Frac.)
pub const FLOAT_PARSE_TOLERANCE: f64 = 0.05 / DENOM as f64;

/// A fraction with a hardcoded denominator [`DENOM`]. Used to ensure numerical stability and
/// eliminate rounding errors.
#[derive(Clone, Copy, Hash, Eq, PartialEq, Ord, PartialOrd, Default)]
pub struct Frac {
    /// The numerator.
    pub numerator: BaseInt,
}

#[derive(Debug, Clone, Error)]
pub enum FracError {
    #[error(
        "Could not convert {0} to a fraction with denominator {}: outside of tolerance {}",
        DENOM,
        FLOAT_PARSE_TOLERANCE
    )]
    FloatParseError(f64),
    #[error("Could not parse {0}")]
    StringParseError(String),
}

impl Frac {
    /// Creates a new `Frac` with the given numerator.
    pub const fn new_with_numerator(numerator: BaseInt) -> Self {
        Self { numerator }
    }

    /// Attempts to read a float as a [`Frac`]. If the float is not within [`FLOAT_PARSE_TOLERANCE`]
    /// of a valid [`Frac`], errors.
    pub fn try_from_float<T: Float>(x: T) -> Result<Self, FracError> {
        let float_num = x.to_f64().unwrap() * DENOM as f64;
        let frac_err = (float_num - float_num.round()).abs();
        if frac_err > FLOAT_PARSE_TOLERANCE {
            Err(FracError::FloatParseError(x.to_f64().unwrap()))
        } else {
            Ok(Self::new_with_numerator(float_num.round() as BaseInt))
        }
    }

    /// Attempts to read a float, panicking if the float is invalid. See [`Frac::try_from_float`] for specifics.
    pub fn from_f64_unchecked(x: f64) -> Self {
        Self::try_from_float(x).unwrap()
    }

    /// Modulo 1: returns the fraction in [0, 1) that is an integer apart from this one.
    pub fn modulo_one(&self) -> Self {
        Self {
            numerator: self.numerator.rem_euclid(Self::DENOM),
        }
    }

    pub const ONE_HALF: Frac = Frac {
        numerator: DENOM / 2,
    };

    pub const ONE: Frac = Frac { numerator: DENOM };

    pub const ZERO: Frac = Frac { numerator: 0 };

    pub const NEG_ONE: Frac = Frac { numerator: -DENOM };

    pub const DENOM: BaseInt = DENOM;
}

impl From<Frac> for f64 {
    fn from(value: Frac) -> Self {
        (value.numerator as f64) / DENOM as f64
    }
}

impl From<Frac> for f32 {
    fn from(value: Frac) -> Self {
        (value.numerator as f32) / DENOM as f32
    }
}

impl FromStr for Frac {
    type Err = FracError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r1 = BaseInt::from_str(s)
            .map(Self::from)
            .map_err(|_e| FracError::StringParseError(s.to_owned()));
        let r2 = f64::from_str(s)
            .ok()
            .and_then(|f| Self::try_from_float(f).ok())
            .ok_or(FracError::StringParseError(s.to_owned()));
        let ints: Vec<Result<BaseInt, <BaseInt as FromStr>::Err>> =
            s.split('/').map(BaseInt::from_str).collect();
        let r3 = match ints[..] {
            [Ok(n), Ok(d)] => {
                if DENOM % d == 0 {
                    Ok(Self::new_with_numerator(n * DENOM / d))
                } else {
                    Err(FracError::StringParseError(s.to_owned()))
                }
            }
            _ => Err(FracError::StringParseError(s.to_owned())),
        };

        r1.or(r2).or(r3)
    }
}

impl Add for Frac {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new_with_numerator(self.numerator + rhs.numerator)
    }
}

impl AddAssign for Frac {
    fn add_assign(&mut self, rhs: Self) {
        self.numerator += rhs.numerator;
    }
}

impl Sub for Frac {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new_with_numerator(self.numerator - rhs.numerator)
    }
}

impl SubAssign for Frac {
    fn sub_assign(&mut self, rhs: Self) {
        self.numerator -= rhs.numerator;
    }
}

impl Mul for Frac {
    type Output = Self;

    /// Note: this can panic because this result may not be representable using the same
    /// denominator. Use this for multiplying by e.g., -1 or 2, but 5/24 x 7/24 will panic.
    fn mul(self, rhs: Self) -> Self::Output {
        let prod = self.numerator as i64 * rhs.numerator as i64;
        assert!(
            prod % DENOM as i64 == 0,
            "Cannot represent {} * {} with denominator {}",
            self,
            rhs,
            DENOM
        );
        Self::new_with_numerator((prod / DENOM as i64) as BaseInt)
    }
}

impl MulAssign for Frac {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Frac {
    type Output = Self;

    /// Note: this can panic because this result may not be representable using the same
    /// denominator. Use this for dividing by e.g., -1 or 2, but 5/24 / 2 will panic.
    fn div(self, rhs: Self) -> Self::Output {
        assert!(
            (self.numerator * DENOM) % rhs.numerator == 0,
            "Cannot represent {} / {} with denominator {}",
            self,
            rhs,
            DENOM
        );
        Self::new_with_numerator(self.numerator * DENOM / rhs.numerator)
    }
}

impl DivAssign for Frac {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Frac {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new_with_numerator(-self.numerator)
    }
}

impl Rem for Frac {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::new_with_numerator(self.numerator % rhs.numerator)
    }
}

impl RemAssign for Frac {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Zero for Frac {
    fn zero() -> Self {
        Self::new_with_numerator(0)
    }

    fn is_zero(&self) -> bool {
        self.numerator == 0
    }
}

impl One for Frac {
    fn one() -> Self {
        Self::new_with_numerator(DENOM)
    }

    fn is_one(&self) -> bool {
        self.numerator == DENOM
    }
}

impl Num for Frac {
    type FromStrRadixErr = <BaseInt as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        BaseInt::from_str_radix(str, radix).map(Self::from)
    }
}

impl SubsetOf<Frac> for f64 {
    fn to_superset(&self) -> Frac {
        Frac::try_from_float(*self).unwrap()
    }

    fn from_superset_unchecked(element: &Frac) -> Self {
        (*element).into()
    }

    fn is_in_subset(_element: &Frac) -> bool {
        true
    }
}

impl SubsetOf<Frac> for Frac {
    fn to_superset(&self) -> Frac {
        *self
    }

    fn from_superset_unchecked(element: &Frac) -> Self {
        *element
    }

    fn is_in_subset(_element: &Frac) -> bool {
        true
    }
}

impl PrimitiveSimdValue for Frac {}

// https://docs.rs/simba/0.8.1/src/simba/simd/simd_value.rs.html#193-195
impl SimdValue for Frac {
    type Element = Frac;

    type SimdBool = bool;

    #[inline(always)]
    fn lanes() -> usize {
        1
    }

    #[inline(always)]
    fn splat(val: Self::Element) -> Self {
        val
    }

    #[inline(always)]
    fn extract(&self, _i: usize) -> Self::Element {
        *self
    }

    #[inline(always)]
    unsafe fn extract_unchecked(&self, _i: usize) -> Self::Element {
        *self
    }

    #[inline(always)]
    fn replace(&mut self, _i: usize, val: Self::Element) {
        self.numerator = val.numerator;
    }

    #[inline(always)]
    unsafe fn replace_unchecked(&mut self, _i: usize, val: Self::Element) {
        self.numerator = val.numerator;
    }

    #[inline(always)]
    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl Field for Frac {}

// TODO check inlining here

impl FromPrimitive for Frac {
    fn from_i64(n: i64) -> Option<Self> {
        i16::from_i64(n).map(Self::from)
    }

    fn from_u64(n: u64) -> Option<Self> {
        i16::from_u64(n).map(Self::from)
    }
}

impl Signed for Frac {
    fn abs(&self) -> Self {
        Self {
            numerator: self.numerator.abs(),
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        Self {
            numerator: self.numerator.abs_sub(&other.numerator),
        }
    }

    fn signum(&self) -> Self {
        Self::from(self.numerator.signum())
    }

    fn is_positive(&self) -> bool {
        self.numerator.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.numerator.is_negative()
    }
}

impl AbsDiffEq for Frac {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        0.into()
    }

    fn abs_diff_eq(&self, other: &Self, _epsilon: Self::Epsilon) -> bool {
        self == other
    }
}

impl UlpsEq for Frac {
    fn default_max_ulps() -> u32 {
        1
    }

    fn ulps_eq(&self, other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
        self.numerator == other.numerator
    }
}

impl RelativeEq for Frac {
    fn default_max_relative() -> Self::Epsilon {
        1.into()
    }

    fn relative_eq(
        &self,
        other: &Self,
        _epsilon: Self::Epsilon,
        _max_relative: Self::Epsilon,
    ) -> bool {
        self.numerator == other.numerator
    }
}

impl RealField for Frac {
    fn is_sign_positive(&self) -> bool {
        self.numerator > 0
    }

    fn is_sign_negative(&self) -> bool {
        self.numerator <= 0
    }

    fn copysign(self, sign: Self) -> Self {
        if sign.is_sign_positive() {
            self.abs()
        } else {
            -self.abs()
        }
    }

    fn max(self, other: Self) -> Self {
        <Self as Ord>::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        <Self as Ord>::min(self, other)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        <Self as Ord>::clamp(self, min, max)
    }

    fn atan2(self, other: Self) -> Self {
        let x: f64 = self.into();
        let y = other.into();
        Self::try_from_float(x.atan2(y)).unwrap()
    }

    fn min_value() -> Option<Self> {
        Some(Self::new_with_numerator(1))
    }

    fn max_value() -> Option<Self> {
        Some(Self::new_with_numerator(BaseInt::MAX))
    }

    fn pi() -> Self {
        panic!()
    }

    fn two_pi() -> Self {
        panic!()
    }

    fn frac_pi_2() -> Self {
        panic!()
    }

    fn frac_pi_3() -> Self {
        panic!()
    }

    fn frac_pi_4() -> Self {
        panic!()
    }

    fn frac_pi_6() -> Self {
        panic!()
    }

    fn frac_pi_8() -> Self {
        panic!()
    }

    fn frac_1_pi() -> Self {
        panic!()
    }

    fn frac_2_pi() -> Self {
        panic!()
    }

    fn frac_2_sqrt_pi() -> Self {
        panic!()
    }

    fn e() -> Self {
        panic!()
    }

    fn log2_e() -> Self {
        panic!()
    }

    fn log10_e() -> Self {
        panic!()
    }

    fn ln_2() -> Self {
        panic!()
    }

    fn ln_10() -> Self {
        panic!()
    }
}

impl ComplexField for Frac {
    type RealField = Frac;

    #[doc = r" Builds a pure-real complex number from the given value."]
    fn from_real(re: Self::RealField) -> Self {
        re
    }

    #[doc = r" The real part of this complex number."]
    fn real(self) -> Self::RealField {
        self
    }

    #[doc = r" The imaginary part of this complex number."]
    fn imaginary(self) -> Self::RealField {
        Self::zero()
    }

    #[doc = r" The modulus of this complex number."]
    fn modulus(self) -> Self::RealField {
        self
    }

    #[doc = r" The squared modulus of this complex number."]
    fn modulus_squared(self) -> Self::RealField {
        self * self
    }

    #[doc = r" The argument of this complex number."]
    fn argument(self) -> Self::RealField {
        if self >= Self::zero() {
            Self::zero()
        } else {
            Self::pi()
        }
    }

    #[doc = r" The sum of the absolute value of this complex number's real and imaginary part."]
    fn norm1(self) -> Self::RealField {
        self.abs()
    }

    #[doc = r" Multiplies this complex number by `factor`."]
    fn scale(self, factor: Self::RealField) -> Self {
        self * factor
    }

    #[doc = r" Divides this complex number by `factor`."]
    fn unscale(self, factor: Self::RealField) -> Self {
        self / factor
    }

    fn floor(self) -> Self {
        Self {
            numerator: self.numerator - self.numerator % DENOM,
        }
    }

    fn ceil(self) -> Self {
        Self {
            numerator: self.numerator + DENOM - self.numerator % DENOM,
        }
    }

    fn round(self) -> Self {
        Self {
            numerator: self.numerator,
        }
    }

    fn trunc(self) -> Self {
        Self::try_from_float(f64::from(self).trunc()).unwrap()
    }

    fn fract(self) -> Self {
        self - self.trunc()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        (self * a) + b
    }

    #[doc = r" The absolute value of this complex number: `self / self.signum()`."]
    #[doc = r""]
    #[doc = r" This is equivalent to `self.modulus()`."]
    fn abs(self) -> Self::RealField {
        Self {
            numerator: self.numerator.abs(),
        }
    }

    #[doc = r" Computes (self.conjugate() * self + other.conjugate() * other).sqrt()"]
    fn hypot(self, other: Self) -> Self::RealField {
        (self.conjugate() * self + other.conjugate() * other).sqrt()
    }

    fn recip(self) -> Self {
        Self::try_from_float(f64::from(self).recip()).unwrap()
    }

    fn conjugate(self) -> Self {
        self
    }

    // TODO macros for this

    fn sin(self) -> Self {
        Self::try_from_float(f64::from(self).sin()).unwrap()
    }

    fn cos(self) -> Self {
        Self::try_from_float(f64::from(self).cos()).unwrap()
    }

    fn sin_cos(self) -> (Self, Self) {
        let (a, b) = f64::from(self).sin_cos();
        (
            Self::try_from_float(a).unwrap(),
            Self::try_from_float(b).unwrap(),
        )
    }

    fn tan(self) -> Self {
        Self::try_from_float(f64::from(self).tan()).unwrap()
    }

    fn asin(self) -> Self {
        Self::try_from_float(f64::from(self).asin()).unwrap()
    }

    fn acos(self) -> Self {
        Self::try_from_float(f64::from(self).acos()).unwrap()
    }

    fn atan(self) -> Self {
        Self::try_from_float(f64::from(self).atan()).unwrap()
    }

    fn sinh(self) -> Self {
        Self::try_from_float(f64::from(self).sinh()).unwrap()
    }

    fn cosh(self) -> Self {
        Self::try_from_float(f64::from(self).cosh()).unwrap()
    }

    fn tanh(self) -> Self {
        Self::try_from_float(f64::from(self).tanh()).unwrap()
    }

    fn asinh(self) -> Self {
        Self::try_from_float(f64::from(self).asinh()).unwrap()
    }

    fn acosh(self) -> Self {
        Self::try_from_float(f64::from(self).acosh()).unwrap()
    }

    fn atanh(self) -> Self {
        Self::try_from_float(f64::from(self).atanh()).unwrap()
    }

    fn log(self, _base: Self::RealField) -> Self {
        panic!()
    }

    fn log2(self) -> Self {
        panic!()
    }

    fn log10(self) -> Self {
        panic!()
    }

    fn ln(self) -> Self {
        panic!()
    }

    fn ln_1p(self) -> Self {
        panic!()
    }

    fn sqrt(self) -> Self {
        Self::try_from_float(f64::from(self).asin()).unwrap()
    }

    fn exp(self) -> Self {
        Self::try_from_float(f64::from(self).asin()).unwrap()
    }

    fn exp2(self) -> Self {
        Self::try_from_float(f64::from(self).asin()).unwrap()
    }

    fn exp_m1(self) -> Self {
        Self::try_from_float(f64::from(self).asin()).unwrap()
    }

    fn powi(self, n: i32) -> Self {
        Self::try_from_float(f64::from(self).powi(n)).unwrap()
    }

    fn powf(self, n: Self::RealField) -> Self {
        Self::try_from_float(f64::from(self).powf(n.into())).unwrap()
    }

    fn powc(self, n: Self) -> Self {
        Self::try_from_float(f64::from(self).powc(n.into())).unwrap()
    }

    fn cbrt(self) -> Self {
        Self::try_from_float(f64::from(self).cbrt()).unwrap()
    }

    fn is_finite(&self) -> bool {
        true
    }

    fn try_sqrt(self) -> Option<Self> {
        f64::from(self)
            .try_sqrt()
            .map(Self::try_from_float)
            .map(Result::unwrap)
    }
}

impl<T: PrimInt> Add<T> for Frac {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::new_with_numerator(self.numerator + DENOM * rhs.to_i16().unwrap())
    }
}

impl<T: PrimInt> Sub<T> for Frac {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::new_with_numerator(self.numerator - DENOM * rhs.to_i16().unwrap())
    }
}

impl<T: PrimInt> Mul<T> for Frac {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new_with_numerator(self.numerator * rhs.to_i16().unwrap())
    }
}

impl<T: PrimInt> From<T> for Frac {
    fn from(x: T) -> Self {
        Self::new_with_numerator(DENOM * x.to_i16().unwrap())
    }
}

impl<T: PrimInt> Rem<T> for Frac {
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        self % Self::from(rhs)
    }
}

impl Frac {
    pub const fn gcd(p: BaseInt, q: BaseInt) -> BaseInt {
        let mut p = p;
        let mut q = q;
        while q != 0 {
            let r = p % q;
            p = q;
            q = r;
        }

        p
    }
}

impl RenderBlocks for Frac {
    fn components(&self) -> Vec<Block> {
        if self.numerator == 0 {
            return vec![Block::new_int(0)];
        }
        let p = self.numerator.abs();
        let d = Self::gcd(p, Self::DENOM);
        let num = Block::new_int(self.numerator as i64 / d as i64);
        let denom = Self::DENOM as u64 / d as u64;
        if denom == 1 {
            vec![num]
        } else {
            vec![Block::Fraction(num.into(), Block::new_uint(denom).into())]
        }
    }
}

impl std::fmt::Debug for Frac {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "frac!({:02}/{})", self.numerator, DENOM)
    }
}

impl Display for Frac {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", DISPLAY.render_to_string(self))
    }
}

#[macro_export]
macro_rules! frac {
    ($num:literal / $denom:expr) => {{
        let d = $denom;
        let n = $num;

        // n / d = x / DENOM
        // DENOM * n / d = x
        if ($crate::fract::Frac::DENOM * n) % d == 0 {
            $crate::fract::Frac::new_with_numerator(($crate::fract::Frac::DENOM * n) / d)
        } else {
            panic!(
                "Invalid fraction: {}/{} cannot be represented as n/{}",
                n,
                d,
                $crate::fract::Frac::DENOM
            )
        }
    }};
    ($num:expr) => {
        $crate::fract::Frac::new_with_numerator(($num as i16) * $crate::fract::Frac::DENOM)
    };
}

#[cfg(test)]
mod tests {
    use crate::markup::UNICODE;

    use super::*;
    use crate::frac;

    use pretty_assertions::assert_eq;

    #[test]
    fn test_macro() {
        assert_eq!(Frac::new_with_numerator(4), frac!(1 / 6));
        assert_eq!(Frac::new_with_numerator(48), frac!(2));
        assert_eq!(Frac::new_with_numerator(4), frac!(1 / 2 + 4));
        assert_eq!(Frac::new_with_numerator(3), frac!(10 / 2.pow(3) * 10));
        assert_eq!(Frac::new_with_numerator(6), frac!(1 / 4));
    }

    #[test]
    fn test_gcd() {
        assert_eq!(Frac::gcd(4, 24), 4);
        assert_eq!(Frac::gcd(25, 4), 1);
        assert_eq!(Frac::gcd(25, 10), 5);
        assert_eq!(Frac::gcd(64, 8), 8);
    }

    // #[test]
    // fn test_ita_display() {
    //     assert_eq!(format!("{}", Frac::new_with_numerator(0)), "0");
    //     assert_eq!(format!("{}", Frac::new_with_numerator(-24)), "1̅");
    //     assert_eq!(format!("{}", Frac::new_with_numerator(-9)), "3̅⁄8");
    //     assert_eq!(format!("{}", Frac::new_with_numerator(-25)), "2̅5̅⁄24");
    //     assert_eq!(format!("{}", Frac::new_with_numerator(-22)), "1̅1̅⁄12");
    // }

    #[test]
    fn test_unic_display() {
        assert_eq!(
            UNICODE
                .render_to_string(&Frac::new_with_numerator(0))
                .as_str(),
            "0"
        );
        assert_eq!(
            UNICODE
                .render_to_string(&Frac::new_with_numerator(-24))
                .as_str(),
            "−1"
        );
        assert_eq!(
            UNICODE
                .render_to_string(&Frac::new_with_numerator(-9))
                .as_str(),
            "−3⁄8"
        );
        assert_eq!(
            UNICODE
                .render_to_string(&Frac::new_with_numerator(-25))
                .as_str(),
            "−25⁄24"
        );
        assert_eq!(
            UNICODE
                .render_to_string(&Frac::new_with_numerator(-22))
                .as_str(),
            "−11⁄12"
        );
    }

    #[test]
    fn test_nalg() {
        use nalgebra as na;
        use nalgebra::Matrix2;

        let m1: Matrix2<Frac> = Matrix2::identity();

        assert_eq!(na::Cholesky::new(m1).unwrap().inverse(), m1);
        assert_eq!(m1.pow(2), m1);

        let m2 = m1.scale(2.into());
        let m3 = m1.scale(Frac::from(2).recip());

        assert_eq!(m2 * m3, Matrix2::identity());
    }

    #[test]
    fn test_mod_one() {
        assert_eq!(frac!(1 / 2).modulo_one(), frac!(1 / 2));
        assert_eq!(frac!(-1 / 2).modulo_one(), frac!(1 / 2));
        assert_eq!(frac!(-3 / 2).modulo_one(), frac!(1 / 2));
        assert_eq!(frac!(1).modulo_one(), frac!(0));
        assert_eq!(frac!(0).modulo_one(), frac!(0));
        assert_eq!(frac!(1 / 3).modulo_one(), frac!(1 / 3));
        assert_eq!(frac!(25 / 3).modulo_one(), frac!(1 / 3));
        assert_eq!(frac!(-35 / 3).modulo_one(), frac!(1 / 3));
    }

    #[test]
    fn test_from_str() {
        assert_eq!(
            Frac::from_str("1/2").unwrap(),
            Frac::new_with_numerator(DENOM / 2)
        );
        assert_eq!(
            Frac::from_str("-5/8").unwrap(),
            Frac::new_with_numerator(-15)
        );
        assert_eq!(
            Frac::from_str("4").unwrap(),
            Frac::new_with_numerator(4 * DENOM)
        );
        assert_eq!(
            Frac::from_str("0.25").unwrap(),
            Frac::new_with_numerator(DENOM / 4)
        );
        assert_eq!(
            Frac::from_str("-0.24999").unwrap(),
            Frac::new_with_numerator(-DENOM / 4)
        );
        assert!(Frac::from_str("0.26").is_err());
    }
}
