//! Lattices in Curium. There is a similar dichotomy to the [`crate::symmop::SymmOp`] and
//! [`crate::isometry::Isometry`] objects: a [`BravaisLattice`] is the idealized geometric group,
//! with a defined geometric type and associated symmetry properties, and that lattice can in turn
//! be represented in a [`LatticeSetting`] which is a particular realization of the lattice as a set
//! of basis vectors. For example, a face-centered cubic [`BravaisLattice`] with the same total
//! volume is fundamentally the same symmetry and same object regardless of which order the basis
//! vectors are given, even if the coordinates used to describe an object differ by setting.

use crate::{
    algebra::GroupElement,
    frac,
    group_classes::CrystalFamily,
    isometry::Isometry,
    symmop::SymmOp,
    units::{angstrom, degree, radian, Angle, Length, Volume},
};
use nalgebra::{Matrix3, Matrix3x1, Matrix4, Vector3};
use simba::scalar::SupersetOf;
use thiserror::Error;

#[cfg(test)]
use proptest_derive::Arbitrary;

impl CrystalFamily {
    /// Gets the letter describing each crystal family, as given in Table 2.1.1.1 of ITA.
    pub fn letter(&self) -> char {
        match *self {
            CrystalFamily::Triclinic => 'a',
            CrystalFamily::Monoclinic => 'm',
            CrystalFamily::Orthorhombic => 'o',
            CrystalFamily::Tetragonal => 't',
            CrystalFamily::Hexagonal => 'h',
            CrystalFamily::Cubic => 'c',
        }
    }
}

/// A lattice system, as defined in 2.1.1.1 of ITA. Describes the point symmetry of the lattice.
/// Differs from [`CrystalFamily`] in the classification of hexagonal-family groups: crystal systems
/// are based on the number (3 = trigonal, 6 = hexagonal), here it's based on the symmetry of the
/// lattice.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(Arbitrary))]
pub enum LatticeSystem {
    Triclinic,
    Monoclinic,
    Orthorhombic,
    Tetragonal,
    Hexagonal,
    Rhombohedral,
    Cubic,
}

impl LatticeSystem {
    /// Gets the crystal family. This converts `Rhombohedral` to `Hexagonal` and leaves everything
    /// else unchanged.
    pub fn family(&self) -> CrystalFamily {
        match *self {
            Self::Triclinic => CrystalFamily::Triclinic,
            Self::Monoclinic => CrystalFamily::Monoclinic,
            Self::Orthorhombic => CrystalFamily::Orthorhombic,
            Self::Tetragonal => CrystalFamily::Tetragonal,
            Self::Hexagonal | Self::Rhombohedral => CrystalFamily::Hexagonal,
            Self::Cubic => CrystalFamily::Cubic,
        }
    }
}

/// The type of translations allowed for the lattice. Defines a Bravais lattice when combined with a
/// lattice system. `SettingDependent`, denoted `S`, indicates A, B, C, or I depending on the group:
/// these are equivalent up to labeling the axes.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(Arbitrary))]
pub enum CenteringType {
    Primitive,
    BodyCentered,
    SettingDependent,
    ACentered,
    BCentered,
    CCentered,
    FaceCentered,
    Rhombohedral,
}

impl CenteringType {
    /// Gets the letter describing each centering type, as given in Table 2.1.1.2 of ITA.
    pub fn letter(&self) -> char {
        match *self {
            CenteringType::Primitive => 'P',
            CenteringType::BodyCentered => 'I',
            CenteringType::SettingDependent => 'S',
            CenteringType::ACentered => 'A',
            CenteringType::BCentered => 'B',
            CenteringType::CCentered => 'C',
            CenteringType::FaceCentered => 'F',
            CenteringType::Rhombohedral => 'R',
        }
    }

    /// Gets the generators of the translational subgroup: the set of translation vectors that
    /// define the centering options. Includes the three basis vectors that shift by entire unit
    /// cells. Fails for `SettingDependent`.
    pub fn centering_ops(&self) -> Option<Vec<SymmOp>> {
        let mut translations = vec![Vector3::x(), Vector3::y(), Vector3::z()];

        let a = Vector3::new(frac!(0), frac!(1 / 2), frac!(1 / 2));
        let b = Vector3::new(frac!(1 / 2), frac!(0), frac!(1 / 2));
        let c = Vector3::new(frac!(1 / 2), frac!(1 / 2), frac!(0));

        // Table 2.1.1.2 of ITA
        translations.extend(match self {
            CenteringType::Primitive => {
                vec![]
            }
            CenteringType::BodyCentered => vec![Vector3::new(frac!(1 / 2), frac!(1 / 2), frac!(0))],
            CenteringType::SettingDependent => {
                return None;
            }
            CenteringType::ACentered => vec![a],
            CenteringType::BCentered => vec![b],
            CenteringType::CCentered => vec![c],
            CenteringType::FaceCentered => vec![a, b, c],
            // 'obverse' setting: this should be part of a setting setting when we have that
            CenteringType::Rhombohedral => vec![
                Vector3::new(frac!(2 / 3), frac!(1 / 3), frac!(1 / 3)),
                Vector3::new(frac!(1 / 3), frac!(2 / 3), frac!(2 / 3)),
            ],
        });

        Some(
            translations
                .into_iter()
                .map(SymmOp::Translation)
                .collect::<Vec<SymmOp>>(),
        )
    }
}

#[allow(non_camel_case_types)]
/// The Bravais lattice classes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Arbitrary))]
pub enum BravaisLatticeType {
    aP,
    mP,
    mS,
    oP,
    oS,
    oI,
    oF,
    tP,
    tI,
    hR,
    hP,
    cP,
    cI,
    cF,
}

impl BravaisLatticeType {
    /// Gets the centering type, which descries the translations that leave the lattice unchanged.
    pub fn centering(&self) -> CenteringType {
        match *self {
            Self::aP | Self::mP | Self::oP | Self::tP | Self::hP | Self::cP => {
                CenteringType::Primitive
            }
            Self::mS | Self::oS => CenteringType::SettingDependent,
            Self::oI | Self::tI | Self::cI => CenteringType::BodyCentered,
            Self::oF | Self::cF => CenteringType::FaceCentered,
            Self::hR => CenteringType::Rhombohedral,
        }
    }

    /// Gets the lattice system, which describes the point group that leaves the lattice unchanged.
    /// This means rhombohedral and hexagonal lattices are treated differently: one has a sixfold
    /// rotation, the other does not.
    pub fn lattice_system(&self) -> LatticeSystem {
        match &self {
            Self::aP => LatticeSystem::Triclinic,
            Self::mP | Self::mS => LatticeSystem::Monoclinic,
            Self::oP | Self::oS | Self::oI | Self::oF => LatticeSystem::Orthorhombic,
            Self::tP | Self::tI => LatticeSystem::Tetragonal,
            Self::hR => LatticeSystem::Rhombohedral,
            Self::hP => LatticeSystem::Hexagonal,
            Self::cP | Self::cI | Self::cF => LatticeSystem::Cubic,
        }
    }

    /// Constructs from family and centering type, if such a type exists, otherwise `None`. Consult
    /// Table 2.1.1.1 of ITA for more information.
    pub fn from_family_and_centering(sys: CrystalFamily, center: CenteringType) -> Option<Self> {
        match (sys, center) {
            (CrystalFamily::Triclinic, CenteringType::Primitive) => Some(Self::aP),
            (CrystalFamily::Monoclinic, CenteringType::Primitive) => Some(Self::mP),
            (
                CrystalFamily::Monoclinic,
                CenteringType::SettingDependent
                | CenteringType::ACentered
                | CenteringType::BCentered
                | CenteringType::BodyCentered
                | CenteringType::CCentered,
            ) => Some(Self::mS),
            (CrystalFamily::Orthorhombic, CenteringType::Primitive) => Some(Self::oP),
            (CrystalFamily::Orthorhombic, CenteringType::BodyCentered) => Some(Self::oI),
            (
                CrystalFamily::Orthorhombic,
                CenteringType::SettingDependent
                | CenteringType::ACentered
                | CenteringType::BCentered
                | CenteringType::CCentered,
            ) => Some(Self::oS),
            (CrystalFamily::Orthorhombic, CenteringType::FaceCentered) => Some(Self::oF),
            (CrystalFamily::Tetragonal, CenteringType::Primitive) => Some(Self::tP),
            (CrystalFamily::Tetragonal, CenteringType::BodyCentered) => Some(Self::tI),
            (CrystalFamily::Hexagonal, CenteringType::Primitive) => Some(Self::hP),
            (CrystalFamily::Hexagonal, CenteringType::Rhombohedral) => Some(Self::hR),
            (CrystalFamily::Cubic, CenteringType::Primitive) => Some(Self::cP),
            (CrystalFamily::Cubic, CenteringType::BodyCentered) => Some(Self::cI),
            (CrystalFamily::Cubic, CenteringType::FaceCentered) => Some(Self::cF),
            _ => None,
        }
    }

    /// Gets the crystal family, defined as the superset of systems with the same number of free
    /// parameters. Essentially maps to the first letter of the lattice name.
    pub fn family(&self) -> CrystalFamily {
        self.lattice_system().family()
    }

    /// Gets the ITA name of the lattice type, the same as its name in Rust.
    pub fn ita_name(&self) -> String {
        format!("{}{}", self.family().letter(), self.centering().letter())
    }
}

/// A crystal lattice.

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

        #[test]
        fn test_combo_bravais_name(
            lat in any::<BravaisLatticeType>()
        ) {
            assert_eq!(format!("{:?}", lat), lat.ita_name());
        }

        #[test]
        fn test_bravais_decomposition_roundtrip(
            lat in any::<BravaisLatticeType>()
        ) {
            assert_eq!(Some(lat), BravaisLatticeType::from_family_and_centering(lat.family(), lat.centering()))
        }

        #[test]
        fn test_family_centering_roundtrip(
            fam in any::<CrystalFamily>(),
            cent in any::<CenteringType>()
        ) {
            let new_cent = match (fam, cent) {
                (CrystalFamily::Monoclinic, CenteringType::SettingDependent | CenteringType::CCentered | CenteringType::BCentered | CenteringType::ACentered | CenteringType::BodyCentered) => CenteringType::SettingDependent,
                (CrystalFamily::Orthorhombic, CenteringType::SettingDependent | CenteringType::CCentered | CenteringType::BCentered | CenteringType::ACentered) => CenteringType::SettingDependent,
                (_fam, c) => c
            };

            let lat = BravaisLatticeType::from_family_and_centering(fam, new_cent).unwrap();
            assert_eq!(lat.family(), fam);
            assert_eq!(lat.centering(), new_cent);
        }
    }
}
