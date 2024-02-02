//! Lattices in Curium. There is a similar dichotomy to the [`crate::symmop::SymmOp`] and
//! [`crate::isometry::Isometry`] objects: a [`BravaisLattice`] is the idealized geometric group,
//! with a defined geometric type and associated symmetry properties, and that lattice can in turn
//! be represented in a [`LatticeSetting`] which is a particular realization of the lattice as a set
//! of basis vectors. For example, a face-centered cubic `BravaisLattice` with the same total volume
//! is fundamentally the same regardless of which order the basis vectors are given.

use crate::{
    isometry::Isometry,
    units::{angstrom, degree, radian, Angle, Length, Volume},
};
use nalgebra::{Matrix3, Matrix3x1, Matrix4, Vector3};
use simba::scalar::SupersetOf;
use thiserror::Error;

/// An idealized lattice: an infinitely repeating set of points in 3D space. For any such set of
/// points, there are many ways to describe the set using a basis: one such choice is a
/// [`LatticeSetting`], not a [`BravaisLattice`]. The Bravais lattice groups can be understood as
/// characterizing the translational subgroup of a crystal: every set of translations that keeps the
/// repeating structure invariant is characterized by the Bravais lattice.
#[derive(Debug, Clone, PartialEq)]
pub enum BravaisLattice {}

/// A particular manifestation of a lattice in 3D coordinates. Essentially a basis in 3D space.
#[derive(Debug, Clone, PartialEq)]
pub struct LatticeSetting {
    m: Matrix3<f64>,
}

#[derive(Debug, Error, Clone)]
pub enum LatticeError {
    #[error("Angles cannot be satisfied: {0}, {1}, {2}")]
    InvalidAngles(f64, f64, f64),
}

impl LatticeSetting {
    /// A new lattice setting from a matrix. Assumes units are in angstroms.
    pub fn new(m: Matrix3<f64>) -> Self {
        Self { m }
    }

    /// Initializes from a, b, c, α, β, γ. Out of all potential rotations of a lattice that satisfy
    /// the parameters, we choose the one that is upper triangular, with no nonzero entries below
    /// the main diagonal.  [AFlow](https://aflow.org/prototype-encyclopedia/triclinic_lattice.html)
    /// gives the formulae. Note that this is *not* the conventional choice as ITA gives: use
    /// [`BravaisLattice`] when you want total control over the representation.
    ///
    /// This is fallible: [not all combinations give valid unit
    /// cells.](https://journals.iucr.org/a/issues/2011/01/00/au5114/au5114.pdf).
    ///
    /// Assumes inputs are in angstroms and degrees.
    pub fn try_from_parameters(
        a: f64,
        b: f64,
        c: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<Self, LatticeError> {
        // https://journals.iucr.org/a/issues/2011/01/00/au5114/au5114.pdf
        let (alpha, beta, gamma) = (alpha % 360., beta % 360., gamma % 360.);
        for expr in &[
            alpha + beta + gamma,
            alpha + beta - gamma,
            alpha - beta + gamma,
            -alpha + beta + gamma,
        ] {
            if !(0f64..360f64).contains(expr) {
                return Err(LatticeError::InvalidAngles(alpha, beta, gamma));
            }
        }
        let (alpha, beta, gamma) = (alpha.to_radians(), beta.to_radians(), gamma.to_radians());

        let (_sin_a, cos_a) = alpha.sin_cos();
        let (_sin_b, cos_b) = beta.sin_cos();
        let (sin_y, cos_y) = gamma.sin_cos();

        let a_vec = Vector3::x().scale(a);
        let b_vec = Vector3::new(cos_y, sin_y, 0.).scale(b);
        let cx = cos_b;
        // println!("({cos_a:.3} - {cos_b:.3} * {cos_y:.3}) / {sin_y:.3}");
        let cy = (cos_a - cos_b * cos_y) / sin_y;
        // dbg!(cx * cx + cy * cy);
        let c_vec = Vector3::new(cx, cy, (1. - cx * cx - cy * cy).sqrt()).scale(c);

        Ok(Self::new(Matrix3::from_columns(&[a_vec, b_vec, c_vec])))
    }

    /// Gets the first basis vector.
    pub fn a_vec(&self) -> Matrix3x1<f64> {
        self.m.column(0).into()
    }

    /// Gets the second basis vector.
    pub fn b_vec(&self) -> Matrix3x1<f64> {
        self.m.column(1).into()
    }

    /// Gets the third basis vector.
    pub fn c_vec(&self) -> Matrix3x1<f64> {
        self.m.column(2).into()
    }

    /// Gets the first basis vector length.
    pub fn a(&self) -> Length {
        Length::new::<angstrom>(self.a_vec().norm())
    }

    /// Gets the second basis vector.
    pub fn b(&self) -> Length {
        Length::new::<angstrom>(self.b_vec().norm())
    }

    /// Gets the third basis vector.
    pub fn c(&self) -> Length {
        Length::new::<angstrom>(self.c_vec().norm())
    }

    /// Gets the angle between b and c, called α.
    pub fn alpha(&self) -> Angle {
        Angle::new::<radian>(self.b_vec().angle(&self.c_vec()))
    }

    /// Gets the angle between c and a, called β.
    pub fn beta(&self) -> Angle {
        Angle::new::<radian>(self.c_vec().angle(&self.a_vec()))
    }

    /// Gets the angle between a and b, called γ.
    pub fn gamma(&self) -> Angle {
        Angle::new::<radian>(self.a_vec().angle(&self.b_vec()))
    }

    /// Gets the parameters (a, b, c, α, β, γ), in units.
    pub fn params(&self) -> (Length, Length, Length, Angle, Angle, Angle) {
        (
            self.a(),
            self.b(),
            self.c(),
            self.alpha(),
            self.beta(),
            self.gamma(),
        )
    }

    /// Returns the metric tensor as a 3x3 matrix. For two row vectors v, w in lattice space (i.e.,
    /// fractional coordinates), the distance in Cartesian coordinates is given by v * G * w^T,
    /// where G is the metric tensor. Thus the metric tensor encodes all necessary information for
    /// distance calculation, and two lattices with the same metric tensor are equivalent up to an
    /// isometry.
    pub fn metric_tensor(&self) -> Matrix3<f64> {
        self.m * self.m.transpose()
    }

    /// Returns the volume of the unit cell.
    pub fn volume(&self) -> Volume {
        let ca = self.alpha().cos().value;
        let cb = self.beta().cos().value;
        let cy = self.gamma().cos().value;
        let scaling = 1.0 - ca * ca - cb * cb - cy * cy + 2.0 * ca * cb * cy;
        self.a() * self.b() * self.c() * scaling.sqrt()
    }

    /// Returns the inverse matrix of the lattice, which maps from Cartesian coordinates to
    /// fractional coordinates.
    pub fn inv_m(&self) -> Matrix3<f64> {
        self.m.try_inverse().unwrap()
    }

    /// Takes an `Isometry` applied to fractional coordinates and conjugates it to work with
    /// Cartesian coordinates. For example, in hexagonal coordinates the triplet `-y, x - y, -z` is
    /// a sixfold rotation, but applying this directly to Cartesian coordinates doesn't work.
    /// Because hexagonal coordinates need, for example, sqrt(3)/2, this cannot be done within
    /// `Frac`. Returns an operation as an affine matrix.
    pub fn to_cartesian_op(&self, iso: Isometry) -> Matrix4<f64> {
        let float_iso_m: Matrix4<f64> = iso.mat().to_subset_unchecked();
        self.m.to_homogeneous() * float_iso_m * self.inv_m().to_homogeneous()
    }
}

#[cfg(test)]
mod tests {
    use uom::marker::{Add, Sub};

    use super::*;
    use crate::units::Quantity;
    use proptest::prelude::*;
    use uom::si::{Dimension, Units};

    fn assert_close<D: Dimension + ?Sized, U: Units<f64> + ?Sized>(
        lhs: Quantity<D, U, f64>,
        rhs: Quantity<D, U, f64>,
    ) where
        <D as uom::si::Dimension>::Kind: Sub + Add,
    {
        let rel_diff = (lhs - rhs).value.abs() / ((lhs + rhs).value.abs() + 1e-9);
        assert!(rel_diff <= 1e-5, "{:?} {:?}", lhs, rhs);
    }

    fn test_single_param_roundtrip(a: f64, b: f64, c: f64, alpha: f64, beta: f64, gamma: f64) {
        let lat_res = LatticeSetting::try_from_parameters(a, b, c, alpha, beta, gamma);
        if let Ok(lat) = lat_res {
            let (a1, b1, c1, alpha1, beta1, gamma1) = lat.params();
            assert_close(a1, Length::new::<angstrom>(a));
            assert_close(b1, Length::new::<angstrom>(b));
            assert_close(c1, Length::new::<angstrom>(c));
            assert_close(alpha1, Angle::new::<degree>(alpha));
            assert_close(beta1, Angle::new::<degree>(beta));
            assert_close(gamma1, Angle::new::<degree>(gamma));
        } else {
            panic!("{} {} {}", alpha, beta, gamma);
        }
    }

    #[test]
    fn test_simple_param_roundtrip() {
        test_single_param_roundtrip(0.1, 0.2, 0.3, 40., 20., 50.);
    }

    prop_compose! {
        fn valid_gamma()(
            alpha in -180f64..180f64,
            beta in -180f64..180f64,
            gamma_coef in 0f64..100f64) -> (f64, f64, f64) {
            // with a, b, y for alpha, beta, gamma
            // y > -(a + b)
            // y > a + b - 360
            // y > b - a
            // y > a - b

            // y < 360 - (a + b)
            // y < a + b
            // y < 360 - a + b
            // y < 360 - b + a
            let (a, b) = (alpha.abs() % 180., beta.abs() % 180.);
            let lb = vec![
                -(a + b),
                a + b - 360.,
                a - b,
                b - a,
            ].into_iter().reduce(|acc, b| f64::max(acc, b)).unwrap();

            let ub = vec![
                360. - (a + b),
                a + b,
                360. - a + b,
                360. + a - b,
            ].into_iter().reduce(|acc, b| f64::min(acc, b)).unwrap();

            // println!("{lb} <= y <= {ub}");
            let y = lb + (gamma_coef / 100.) * (ub - lb);

            (a, b, y)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]
        #[test]
        fn test_param_roundtrip(
            a in 0.1f64..=20f64,
            b in 0.1f64..=20f64,
            c in 0.1f64..=20f64,
            aby in valid_gamma()
        ) {
            let (alpha, beta, gamma) = aby;
            for expr in &[
            alpha + beta + gamma,
            alpha + beta - gamma,
            alpha - beta + gamma,
            -alpha + beta + gamma,
            ] {
            prop_assume!((0f64..360f64).contains(expr));
            }
            test_single_param_roundtrip(a, b, c, alpha, beta, gamma)
        }
    }
}
