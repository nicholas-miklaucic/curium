//! Defines the symmetry operations in 3D space.

use std::str::FromStr;

use nalgebra::{Matrix3, Matrix4, RowVector4, Vector3};
use thiserror::Error;

use crate::frac::Frac;

#[derive(Debug, Error, Clone)]
pub enum SymmOpError {
    #[error("Matrix not affine: {0}")]
    NotHomogenous(Matrix4<Frac>),
}

/// A symmetry operation in 3D space.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct SymmOp {
    /// The rotation matrix. Can have determinant 1 or -1.
    pub rot: Matrix3<Frac>,
    /// The translation vector.
    pub tau: Vector3<Frac>,
}

impl SymmOp {
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
        mat.column_part_mut(0, 3).copy_from(&self.tau);
        mat
    }
}

impl TryFrom<Matrix4<Frac>> for SymmOp {
    type Error = SymmOpError;

    /// Tries to convert from an affine matrix. If the matrix is not affine, returns an error.
    fn try_from(mat: Matrix4<Frac>) -> Result<Self, Self::Error> {
        if mat.row(3) != Matrix4::identity().row(3) {
            Err(SymmOpError::NotHomogenous(mat))
        } else {
            Ok(Self::new(
                mat.fixed_view::<3, 3>(0, 0).into_owned(),
                mat.fixed_view::<3, 1>(0, 3).into_owned(),
            ))
        }
    }
}

/// Parses a coordinate, e.g., `x - y`, from a String.
fn parse_coord(s: &str) -> Result<RowVector4<Frac>, SymmOpError> {}

impl FromStr for SymmOp {
    type Err = SymmOpError;

    /// Parses a symmetry operation from a triplet, e.g., `-y, x-y, z+1/3`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut lines = s.lines();
        let rot = Matrix3::from_str(lines.next().unwrap())?;
        let tau = Vector3::from_str(lines.next().unwrap())?;
        Ok(Self::new(rot, tau))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = SymmOp::identity();
        assert_eq!(id.rot, Matrix3::identity());
        assert_eq!(id.tau, Vector3::zeros());
    }
}
