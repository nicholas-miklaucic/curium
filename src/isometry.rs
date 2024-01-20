//! Defines the symmetry operations in 3D space.

use std::{convert::identity, str::FromStr};

use nalgebra::{Matrix3, Matrix3x4, Matrix4, RowVector4, Vector3};
use thiserror::Error;

use crate::frac::Frac;

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
    /// The rotation matrix. Can have determinant 1 or -1.
    pub rot: Matrix3<Frac>,
    /// The translation vector.
    pub tau: Vector3<Frac>,
}

impl Isometry {
    /// Creates a new symmetry operation.
    pub fn new(rot: Matrix3<Frac>, tau: Vector3<Frac>) -> Self {
        Self { rot, tau }
    }

    /// Creates a new identity symmetry operation.
    pub fn identity() -> Self {
        Self::new(Matrix3::identity(), Vector3::zeros())
    }

    /// Returns the affine matrix representing this operation.
    pub fn affine_matrix(&self) -> Matrix4<Frac> {
        let mut mat = Matrix4::identity();
        mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.rot);
        mat.fixed_view_mut::<3, 1>(0, 3).copy_from(&self.tau);
        mat
    }
}

impl TryFrom<Matrix4<Frac>> for Isometry {
    type Error = IsometryError;

    /// Tries to convert from an affine matrix. If the matrix is not affine, returns an error.
    fn try_from(mat: Matrix4<Frac>) -> Result<Self, Self::Error> {
        if mat.row(3) != Matrix4::identity().row(3) {
            Err(IsometryError::NotHomogenous(mat))
        } else {
            Ok(Self::new(
                mat.fixed_view::<3, 3>(0, 0).into_owned(),
                mat.fixed_view::<3, 1>(0, 3).into_owned(),
            ))
        }
    }
}

/// Parses a single coefficient.
fn parse_coef(s: &str) -> Result<Frac, IsometryError> {
    let s = s.replace(" ", "");
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
    for (i, var) in vec![(0, "x"), (1, "y"), (2, "z"), (3, "v")] {
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
    let s_pm = s.replace("-", "+-").replace(" ", "");
    let coefs = s_pm.split("+").filter(|s| !s.is_empty());
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
            s.split(",").map(parse_coord).collect();
        let rows = rows?;
        if rows.len() != 3 {
            return Err(IsometryError::CoordParse(s.to_owned()));
        } else {
            let m = Matrix3x4::<Frac>::from_rows(&rows[..]);
            Ok(Isometry {
                rot: m.fixed_view::<3, 3>(0, 0).clone_owned(),
                tau: m.column(3).clone_owned(),
            })
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
        assert_eq!(id.rot, Matrix3::identity());
        assert_eq!(id.tau, Vector3::zeros());
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
        let iso1 = Isometry::new(
            Matrix3::from_vec(vec![f0, fm1, f0, f1, fm1, Frac::ONE_HALF, f0, f0, f1]).transpose(),
            Vector3::from_vec(vec![f0, f0, Frac::from_f64_unchecked(2. / 3.)]),
        );
        let iso1_p = "-y, x-y+z/2, z+2/3".parse::<Isometry>().unwrap();
        assert_eq!(
            iso1_p,
            iso1,
            "Parsing failed:\n{}\n{}",
            iso1_p.affine_matrix(),
            iso1.affine_matrix()
        )
    }

    #[test]
    fn test_parse_tau() {
        let iso_p = Isometry::from_str("-y+3/4, -x+1/4, z+1/4").unwrap();
        let iso = Isometry::new(
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
            iso_p.affine_matrix(),
            iso.affine_matrix()
        )
    }
}
