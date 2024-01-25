//! Defines the symmetry operations in 3D space.

use std::{ops::Mul, str::FromStr};

use nalgebra::{Matrix3, Matrix3x4, Matrix4, RowVector4, Translation3, Vector3};
use simba::scalar::SupersetOf;
use thiserror::Error;

use crate::{
    frac::Frac,
    markup::{Block, BlockSequence},
};

#[derive(Debug, Error, Clone)]
pub enum IsometryError {
    #[error("Matrix not affine: {0}")]
    NotHomogenous(Matrix4<Frac>),
    #[error("Cannot parse coordinate: {0}")]
    CoordParse(String),
}

/// A symmetry operation in 3D space represented generically.
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct Isometry {
    /// The 4x4 affine matrix.
    m: Matrix4<Frac>,
}

impl Mul for Isometry {
    type Output = Self;

    fn mul(self, rhs: Isometry) -> Self::Output {
        Self::new_affine(self.mat() * rhs.mat())
    }
}

impl Isometry {
    /// Creates a new symmetry operation from rotation and translation components.
    pub fn new_rot_tau(rot: Matrix3<Frac>, tau: Vector3<Frac>) -> Self {
        Self {
            m: Translation3::new(tau.x, tau.y, tau.z).to_homogeneous() * rot.to_homogeneous(),
        }
    }

    /// Creates a new isometry from an affine matrix. Needs to be valid.
    pub fn new_affine(m: Matrix4<Frac>) -> Self {
        m.try_into().unwrap()
    }

    /// Creates a new identity symmetry operation.
    pub fn identity() -> Self {
        Self::new_rot_tau(Matrix3::identity(), Vector3::zeros())
    }

    /// Returns the affine matrix representing this operation.
    pub fn mat(&self) -> Matrix4<Frac> {
        self.m
    }

    /// Returns the rotation matrix.
    pub fn rot(&self) -> Matrix3<Frac> {
        self.m.fixed_view::<3, 3>(0, 0).clone_owned()
    }

    /// Returns the translation vector.
    pub fn tau(&self) -> Vector3<Frac> {
        self.m.fixed_view::<3, 1>(0, 3).clone_owned()
    }
}

impl Isometry {
    pub fn inv(&self) -> Self {
        // 1.2.2.8 of ITA
        // we could probably implement Cramer's rule, but meh
        let float_m: Matrix4<f64> = self.m.to_subset_unchecked();
        let float_m_inv = float_m.try_inverse().unwrap();
        let m_inv: Matrix4<Frac> =
            Matrix4::from_iterator(float_m_inv.iter().map(|&fl| Frac::from_f64_unchecked(fl)));
        Self::new_affine(m_inv)
    }
}

impl TryFrom<Matrix4<Frac>> for Isometry {
    type Error = IsometryError;

    /// Tries to convert from an affine matrix. If the matrix is not affine, returns an error.
    fn try_from(mat: Matrix4<Frac>) -> Result<Self, Self::Error> {
        if mat.row(3) != Matrix4::identity().row(3) {
            Err(IsometryError::NotHomogenous(mat))
        } else {
            Ok(Self::new_rot_tau(
                mat.fixed_view::<3, 3>(0, 0).into_owned(),
                mat.fixed_view::<3, 1>(0, 3).into_owned(),
            ))
        }
    }
}

/// Parses a single coefficient.
fn parse_coef(s: &str) -> Result<Frac, IsometryError> {
    let s = s.replace(' ', "");
    let s0 = s.replace('v', "");
    let s1 = s.replace('v', "1");
    let res = Frac::from_str(s0.as_str())
        .or(Frac::from_str(s1.as_str()))
        .map_err(|_e| IsometryError::CoordParse(s.to_owned()));

    res
}

fn parse_single_var(s: &str) -> Result<RowVector4<Frac>, IsometryError> {
    let err = IsometryError::CoordParse(s.to_owned());
    let mut coef = Err(err.clone());
    let mut ivals = vec![];
    for (i, var) in [(0, "x"), (1, "y"), (2, "z"), (3, "v")] {
        if i == 3 || s.contains(var) {
            coef = coef.or(parse_coef(s.replace(var, "v").as_str()));
            ivals.push(i);
        }
    }
    let i4 = Matrix4::<Frac>::identity();
    // dbg!(&ivals, s);
    match ivals[..] {
        [i] | [i, 3] => coef.map(|f| i4.row(i).scale(f)),
        _ => Err(err),
    }
}

/// Parses a coordinate, e.g., `x - y`, from a String.
fn parse_coord(s: &str) -> Result<RowVector4<Frac>, IsometryError> {
    let s_pm = s.replace('-', "+-").replace(' ', "");
    let coefs = s_pm.split('+').filter(|s| !s.is_empty());
    // println!("{:?}", coefs.collect::<Vec<_>>());
    // let coefs = s_pm.split("+").filter(|s| !s.is_empty());
    // let vecs = coefs.map(parse_single_var).collect::<Vec<_>>();
    // dbg!(vecs).into_iter().sum()
    coefs.map(parse_single_var).sum()
}

impl FromStr for Isometry {
    type Err = IsometryError;

    /// Parses a symmetry operation from a triplet, e.g., `-y, x-y, z+1/3`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let rows: Result<Vec<RowVector4<Frac>>, IsometryError> =
            s.split(',').map(parse_coord).collect();
        let rows = rows?;
        if rows.len() != 3 {
            Err(IsometryError::CoordParse(s.to_owned()))
        } else {
            let m = Matrix3x4::<Frac>::from_rows(&rows[..]);
            Ok(Isometry::new_rot_tau(
                m.fixed_view::<3, 3>(0, 0).clone_owned(),
                m.column(3).clone_owned(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;
    use crate::frac;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_identity() {
        let id = Isometry::identity();
        assert_eq!(id.rot(), Matrix3::identity());
        assert_eq!(id.tau(), Vector3::zeros());
    }

    #[test]
    fn test_parse() {
        let i4: Matrix4<Frac> = Matrix4::identity();
        assert_eq!(parse_coef("v").unwrap(), Frac::ONE);
        assert_eq!(parse_single_var("x").unwrap(), i4.row(0));
        assert_eq!("x, y, z".parse::<Isometry>().unwrap(), Isometry::identity());
        assert_eq!(parse_coord("-y").unwrap(), -i4.row(1));
        let f1 = Frac::ONE;
        let fm1 = Frac::NEG_ONE;
        let f0 = Frac::ZERO;
        let iso1 = Isometry::new_rot_tau(
            Matrix3::from_vec(vec![f0, fm1, f0, f1, fm1, Frac::ONE_HALF, f0, f0, f1]).transpose(),
            Vector3::from_vec(vec![f0, f0, Frac::from_f64_unchecked(2. / 3.)]),
        );
        let iso1_p = "-y, x-y+z/2, z+2/3".parse::<Isometry>().unwrap();
        assert_eq!(
            iso1_p,
            iso1,
            "Parsing failed:\n{}\n{}",
            iso1_p.mat(),
            iso1.mat()
        )
    }

    #[test]
    fn test_parse_tau() {
        let iso_p = Isometry::from_str("-y+3/4, -x+1/4, z+1/4").unwrap();
        let iso = Isometry::new_rot_tau(
            matrix![
                frac!(0), frac!(-1), frac!(0);
                frac!(-1), frac!(0), frac!(0);
                frac!(0), frac!(0), frac!(1)
            ],
            Vector3::new(frac!(3 / 4), frac!(1 / 4), frac!(1 / 4)),
        );
        dbg!(frac!(1 / 4));
        assert_eq!(
            iso_p,
            iso,
            "Parsing failed:\n{}\n{}",
            iso_p.mat(),
            iso.mat()
        )
    }
}
