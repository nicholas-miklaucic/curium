//! A symmetry operation in 3D space, considered geometrically. Acts on 3D space as an [`Isometry`],
//! but specifically considers the subgroup of isometries that occurs in crystal space groups.

use std::convert::identity;

use nalgebra::{iter, zero, Matrix3, Matrix4, Point3, Vector3};
use num_traits::FromPrimitive;
use thiserror::Error;

use crate::{
    frac::{BaseInt, Frac, DENOM},
    isometry::Isometry,
};

/// The axis of rotation.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct RotationAxis {
    pub x: Frac,
    pub y: Frac,
    pub z: Frac,
}

/// The kind of rotation: sense and order. -2 is a reflection, which we don't represent here.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum RotationKind {
    PosTwo,
    PosThree,
    PosFour,
    PosSix,
    NegThree,
    NegFour,
    NegSix,
}

/// The amount of translation along the axis of rotation in a screw rotation.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct ScrewOrder {
    order: BaseInt,
}

/// A plane (particularly of reflection.)
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct Plane {
    pub x: Frac,
    pub y: Frac,
    pub z: Frac,
    pub d: Frac,
}

/// A symmetry operation. See section 1.2.1 of the International Tables of Crystallography.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum SymmOp {
    /// The identity.
    Identity,
    /// Inversion around a point.
    Inversion(Point3<Frac>),
    /// A translation.
    Translation(Vector3<Frac>),
    /// A proper rotation. Limited to order 2, 3, 4, or 6 in crystal space groups.
    Rotation(RotationAxis, RotationKind),
    /// A screw rotation: rotation and translation along the axis.
    Screw(RotationAxis, RotationKind, ScrewOrder),
    /// A rotoinversion. Rotoinversions of order 2 are isomorphic to reflections and should be
    /// represented as such.
    Rotoinversion(RotationAxis, RotationKind),
    /// A reflection through a plane.
    Reflection(Plane),
    /// A glide reflection: reflection and translation parallel to the reflection plane.
    Glide(Plane, Vector3<Frac>),
}

#[derive(Debug, Clone, Error)]
pub enum SymmOpError {
    #[error("Matrix not symmetry operation: {0:?}")]
    NotSymmetryOperation(Isometry),
}

fn det3x3(m: Matrix3<Frac>) -> Frac {
    let m_nums: Vec<i32> = m.iter().map(|f| f.numerator.into()).collect();
    if let [a, d, g, b, e, h, c, f, i] = m_nums.as_slice() {
        // matrices iterate in column-major order
        // | a b c |
        // | d e f |
        // | g h i |
        // det = a(ei - hf) + b(di - fg) + c(dh - eg)
        // this scales by denom^3 when we multiply each number by denom
        let det_scaled = a * (e * i - h * f) + b * (d * i - g * f) + c * (d * h - g * e);
        let den = DENOM as i32;

        assert!(
            det_scaled % (den * den * den) == 0,
            "Invalid determinant {}! {}",
            m,
            det_scaled
        );

        Frac::new_with_numerator((det_scaled / (den * den)) as i16)
    } else {
        panic!("Matrix3 should always have 9 elements, so something went wrong")
    }
}

impl SymmOp {
    /// Classifies the isometry according to its geometric meaning.
    pub fn classify_affine(m: Isometry) -> Result<Self, SymmOpError> {
        let err = Err(SymmOpError::NotSymmetryOperation(m));
        // c.f. International Tables of Crystallography, section 1.2.2.4
        // (1)
        // (a) (i) Classify according to determinant of W
        let det = det3x3(m.rot);
        let is_proper = if det == Frac::ONE {
            true
        } else if det == Frac::NEG_ONE {
            false
        } else {
            return err;
        };
        // (a) (ii) Classify angle according to trace
        let tr = m.rot.m11 + m.rot.m22 + m.rot.m33;
        let tr_det = tr * det;
        if tr_det.numerator % DENOM != 0 {
            return err;
        }

        let tr_det = tr_det.numerator / DENOM;
        let rot_type = match tr_det {
            3 => 1,
            2 => 6,
            1 => 4,
            0 => 3,
            -1 => 2,
            _ => {
                return err;
            }
        };

        let rot_type = if is_proper { rot_type } else { -rot_type };

        if rot_type == 1 {
            // translation or identity, no rotational component
            if m.tau == Vector3::<Frac>::zeros() {
                return Ok(Self::Identity);
            } else {
                return Ok(Self::Translation(m.tau));
            }
        } else if rot_type == -1 {
            // inversion, find center
            // solve for fixed point to find center
            // -p + τ = p
            // p = τ/2
            return Ok(Self::Inversion(m.tau.scale(Frac::ONE_HALF).into()));
        }

        todo!()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;

    use super::*;

    #[test]
    fn test_det() {
        let m: Matrix3<Frac> = Matrix3::from_vec(vec![
            1.into(),
            0.into(),
            0.into(),
            (-1).into(),
            1.into(),
            1.into(),
            0.into(),
            (-1).into(),
            1.into(),
        ]);
        let rot_float: Matrix3<f64> =
            Matrix3::from_vec(m.iter().map(|f| f64::from(*f)).collect::<Vec<f64>>());

        assert_ulps_eq!(rot_float.determinant(), det3x3(m).into());
    }
}
