//! A symmetry operation in 3D space, considered geometrically. Acts on 3D space as an [`Isometry`],
//! but specifically considers the subgroup of isometries that occurs in crystal space groups.

use std::{convert::identity, iter::successors};

use nalgebra::{
    iter, matrix, zero, Const, Matrix3, Matrix4, OMatrix, Point3, SMatrix, Vector, Vector3,
};
use num_traits::{FromPrimitive, Signed, Zero};
use thiserror::Error;

use crate::frac;
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

impl From<Vector3<Frac>> for RotationAxis {
    fn from(v: Vector3<Frac>) -> Self {
        Self {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

/// The kind of rotation: sense and order.
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

impl RotationKind {
    pub fn new(is_ccw: bool, order: usize) -> Self {
        match (is_ccw, order) {
            (_, 2) => Self::PosTwo,
            (true, 3) => Self::PosThree,
            (false, 3) => Self::NegThree,
            (true, 4) => Self::PosFour,
            (false, 4) => Self::NegFour,
            (true, 6) => Self::PosSix,
            (false, 6) => Self::NegSix,
            _ => panic!("Invalid sense and order: {}, {}", is_ccw, order),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct RotationType {
    /// The sense and order of the rotation.
    kind: RotationKind,
    /// Whether the rotation is proper or improper.
    is_proper: bool,
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

impl Plane {
    pub fn from_basis_and_origin(
        v1: Vector3<Frac>,
        v2: Vector3<Frac>,
        origin: Point3<Frac>,
    ) -> Self {
        let normal = v1.cross(&v2);
        let d = -normal.dot(&origin.coords);
        Self {
            x: normal.x,
            y: normal.y,
            z: normal.z,
            d,
        }
    }
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
    /// A rotation or rotoinversion, potentially with an additional translation (a screw motion).
    /// Rotoinversions of order 2 are isomorphic to reflections and should be represented as such.
    GeneralRotation(RotationAxis, RotationType, Point3<Frac>, Vector3<Frac>),
    /// A reflection through a plane, with optional glide.
    Reflection(Plane, Vector3<Frac>),
}

impl SymmOp {
    pub fn new_generalized_rotation(
        axis: RotationAxis,
        kind: RotationKind,
        is_proper: bool,
        center: Point3<Frac>,
        tau: Vector3<Frac>,
    ) -> Self {
        Self::GeneralRotation(axis, RotationType { kind, is_proper }, center, tau)
    }

    pub fn new_generalized_reflection(plane: Plane, tau: Vector3<Frac>) -> Self {
        Self::Reflection(plane, tau)
    }
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
        // det = a(ei - hf) - b(di - fg) + c(dh - eg)
        // this scales by denom^3 when we multiply each number by denom
        let det_scaled = a * (e * i - h * f) - b * (d * i - g * f) + c * (d * h - g * e);
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
    #[allow(non_snake_case)] // we want to use the same notation as ITA
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

        let (rot_type, order): (BaseInt, BaseInt) = if is_proper {
            (rot_type, rot_type)
        } else {
            // -1 has order 2, -3 has order 6
            (-rot_type, rot_type + rot_type * (rot_type % 2))
        };

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
        } else {
            // (b) find rotation axis

            // Consider Y(W) = W^(k-1) + W^(k-2) + ... + W + I where k is the order of the rotation.
            // Note that W Y(W) = I + W^(k-1) + ... + W^2 + W = Y(W), so Y(W) sends anything not
            // orthogonal to it to the axis. So we can try Y(W) v and optionally try a second v if
            // we get unlucky and Y(W) v = 0. If we have a rotoinversion, then instead of Y(W) we
            // have Y(-W).
            let W = m.rot;
            let k = rot_type.abs();
            let W_abs = W * det;
            let I: Matrix3<Frac> = Matrix3::identity();
            let Y: Matrix3<Frac> = successors(Some(I), |acc| Some(W_abs * acc))
                .take(k as usize)
                .sum();

            // if we make this integral, we make sure the multiplication stays within the Frac field
            // hopefully we never have to deal with the axis 1, -1, 3
            let v1 = Vector3::new(frac!(1), frac!(-1), frac!(3));
            // make sure this isn't parallel to v1 and it'll cover the other cases
            let v2 = Vector3::new(frac!(1), frac!(1), frac!(1));

            let axis_v1 = Y * v1;
            let axis: Vector3<Frac> = if axis_v1.is_zero() {
                // :(
                Y * v2
            } else {
                axis_v1
            };

            // (c) sense of rotation

            // the sense is, where u is the axis, x is some vector not parallel to u, and d = det W

            // | u1 x1 (dWx)1 |
            // | u2 x2 (dWx)2 |
            // | u3 x3 (dWx)3 |

            // We can pick x to be either <1, 0, 0> or <0, 0, 1>, using the second only if the first
            // is parallel to u. If x is <1, 0, 0>, then Wx is the first column of W, and otherwise
            // it's the 3rd. We can use Laplace expansion with x, and in either case we have that
            // the only term has negative sign. If x is <1, 0, 0>, then the determinant is -d(u2 *
            // W_13 - u3 * W_12), and otherwise it's -d(u1 * W_32 - u2 * W_31). Because we're not
            // sure if the intermediate products will work, we take an implicit denominator out of
            // it. We only care about the sign anyway.
            let sense = if axis[2].is_zero() && axis[1].is_zero() {
                // parallel to <1, 0, 0>, use <0, 0, 1>
                det * (axis.y * W.m13.numerator - axis.x * W.m23.numerator)
            } else {
                det * (axis.z * W.m21.numerator - axis.y * W.m31.numerator)
            };

            // println!(
            //     "Z: {}",
            //     matrix![
            //         axis.x, frac!(1), W.m11;
            //         axis.y, frac!(0), W.m21;
            //         axis.z, frac!(0), W.m31
            //     ]
            // );
            // println!("W: {}\naxis: {}\ndet: {}", W, axis, det);

            let sense = sense.numerator.signum();
            // this would mean our x logic above failed
            assert!(sense != 0 || order == 2, "{} is 0???", sense);
            let is_ccw = sense == 1;
            let rot_kind = RotationKind::new(is_ccw, order as usize);

            // (2) Analysis of the translation column τ, which they call w
            let w: Vector3<Frac> = m.tau;

            // (a) Consider the combined operation (W, w). Call the order k. We have that (W, w)^k =
            // (I, w^k). w^k/k is therefore the translation component associated with a single
            // operation. This is 1/k Y(W) w, where k is the order

            // This section is a little unclear on how to handle rotoinversions. I'm using the order
            // of the full operation and not the order of the rotation, which should be correct, but
            // it may be possible to save some multiplications or do something more precise.
            let YW: Matrix3<Frac> = successors(Some(I), |acc| Some(W * acc))
                .take(order as usize)
                .sum();
            let t = YW * w;
            let w_g = t / frac!(order);
            // the location part of the translation, which isn't a screw or a glide
            let w_l = w - w_g;

            // println!("Y(±W): {}", Y);
            // println!("Y(W): {}", YW);
            // println!("w: {}", w);
            // println!("t: {}", t);
            // println!("w_l: {}", w_l);
            // println!("w_g: {}", w_g);

            // (b) Fixed points

            // We want to solve for the fixed points of the reduced operation (without a screw or
            // glide), given by Wf + w_l = f. The tricky bit is that this is usually not going to
            // have just one solution, and we need to cleanly parameterize the subspace of
            // solutions, whatever that is.

            // We first put this into the form (W - I)f = -w_l. We now have a classical matrix
            // equation. We can then put (W - I) in row echelon form. We use an augmented form

            // [W | I | -wl]

            // When we're done, the entries of the zero rows in I give the kernel, and the non-zero
            // rows in -w_l give a first solution.
            let WmI = W - I;
            let aug: SMatrix<Frac, 3, 7> = SMatrix::from_columns(&[
                WmI.column(0),
                WmI.column(1),
                WmI.column(2),
                I.column(0),
                I.column(1),
                I.column(2),
                w_l.column(0),
            ]);
            // println!("aug: {}", aug);
            let mut aug = aug.clone_owned();
            gauss_eliminate(&mut aug);
            // println!("aug eliminated: {}", aug);

            // aug is now in its reduced form. If the first three entries in a row are 0, then we
            // check that the last entry is 0 (otherwise our system is unsolvable!), then we read
            // off a basis vector of the kernel from the second set of three rows. Otherwise, we can
            // solve for that row of the result by dividing out by the nonzero element.
            let mut kernel_basis = vec![];
            let mut center = Point3::origin();
            for row in 0..3 {
                if aug.fixed_view::<1, 3>(row, 0) == Vector3::zero().transpose() {
                    // part of kernel
                    // if this fails, the system has no solutions and we goofed
                    assert!(aug[(row, 6)].is_zero());
                    // add basis vector to kernel basis
                    kernel_basis.push(aug.fixed_view::<1, 3>(row, 3).transpose());
                } else {
                    // part of solution
                    let mut sol_dim = 0;
                    while aug[(row, sol_dim)].is_zero() {
                        sol_dim += 1;
                    }
                    center[sol_dim] = aug[(row, 6)] / aug[(row, sol_dim)];
                }
            }

            // We already covered the cases where the kernel has dimension 3: identity and
            // translation.

            // dbg!(&kernel_basis);
            match kernel_basis[..] {
                [] => {
                    // single solution: rotoinversion center
                    Ok(Self::new_generalized_rotation(
                        axis.into(),
                        rot_kind,
                        is_proper,
                        center,
                        w_l,
                    ))
                }
                [_u] => {
                    // single axis: rotation or screw
                    Ok(Self::new_generalized_rotation(
                        axis.into(),
                        rot_kind,
                        is_proper,
                        center,
                        w_g,
                    ))
                }
                [u, v] => {
                    // two axes: glide
                    Ok(Self::new_generalized_reflection(
                        Plane::from_basis_and_origin(u, v, center),
                        w_g,
                    ))
                }
                _ => Err(SymmOpError::NotSymmetryOperation(m)),
            }
        }
    }
}

/// Applies Gaussian elimination, being careful to keep the answer within `Frac`.
fn gauss_eliminate(aug: &mut OMatrix<Frac, Const<3>, Const<7>>) {
    let mut row = 0;

    /// Make (r_elim, c_elim) 0 by subtracting a multiple of row r.
    fn zero_out(
        aug: &mut OMatrix<Frac, Const<3>, Const<7>>,
        r_elim: usize,
        c_elim: usize,
        r: usize,
    ) {
        if !aug[(r_elim, c_elim)].is_zero() {
            let factor = aug[(r_elim, c_elim)] / aug[(r, c_elim)];
            aug.set_row(r_elim, &(aug.row(r_elim) - aug.row(r).scale(factor)));
        }
    }

    // only need to eliminate twice
    for col in 0..2 {
        // find nonzero entry to put in upper left
        let mut nonzero_row = row;
        while nonzero_row < 3 && aug[(nonzero_row, col)].is_zero() {
            nonzero_row += 1;
        }
        if nonzero_row != 3 {
            aug.swap_rows(row, nonzero_row);
            // eliminate column from remaining rows
            for r_elim in (row + 1)..3 {
                zero_out(aug, r_elim, col, row);
            }
            // this row is now in echelon form
            row += 1;
        }
    }

    // back substitution: we don't want to have to worry too much about going out of `Frac` range,
    // so we aren't going to have the requirement that the diagonal is all-1, but we do want to
    // eliminate the upper triangle.
    if !aug[(2, 2)].is_zero() {
        zero_out(aug, 1, 2, 2);
        zero_out(aug, 0, 2, 2);
    }

    if !aug[(1, 1)].is_zero() {
        zero_out(aug, 0, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use std::str::FromStr;

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

    #[test]
    fn test_id_inv() {
        assert_eq!(
            SymmOp::classify_affine(Isometry::from_str("x,y,z").unwrap()).unwrap(),
            SymmOp::Identity
        );

        assert_eq!(
            SymmOp::classify_affine(Isometry::from_str("-x,-y,-z").unwrap()).unwrap(),
            SymmOp::Inversion(Point3::new(Frac::ZERO, Frac::ZERO, Frac::ZERO))
        );

        assert_eq!(
            SymmOp::classify_affine(Isometry::from_str("1-x,1-y,-z").unwrap()).unwrap(),
            SymmOp::Inversion(Point3::new(Frac::ONE_HALF, Frac::ONE_HALF, Frac::ZERO))
        );
    }

    #[test]
    fn test_glide() {
        let iso = Isometry::from_str("-y+3/4, -x+1/4, z+1/4").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        assert_eq!(
            symm,
            SymmOp::Reflection(
                Plane::from_basis_and_origin(
                    Vector3::new(frac!(1), frac!(-1), frac!(0)),
                    Vector3::new(frac!(0), frac!(0), frac!(1)),
                    Point3::new(frac!(1 / 2), frac!(0), frac!(0))
                ),
                Vector3::new(frac!(1 / 4), frac!(-1 / 4), frac!(1 / 4))
            )
        );
    }

    #[test]
    fn test_rotoinversion() {
        let iso = Isometry::from_str("y+1/4, -x+1/4, z+3/4").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        assert_eq!(
            symm,
            SymmOp::new_generalized_rotation(
                RotationAxis {
                    x: frac!(0),
                    y: frac!(0),
                    z: frac!(1)
                },
                RotationKind::NegFour,
                true,
                Point3::new(frac!(1 / 4), frac!(0), frac!(0)),
                Vector3::new(frac!(0), frac!(0), frac!(3 / 4))
            )
        );
    }

    #[test]
    fn test_row_reduce() {
        let mut aug1: SMatrix<Frac, 3, 7> = SMatrix::identity();
        aug1.set_column(0, &Vector3::new(frac!(2), frac!(-3), frac!(-2)));
        aug1.set_column(1, &Vector3::new(frac!(1), frac!(-1), frac!(1)));
        aug1.set_column(2, &Vector3::new(frac!(-1), frac!(2), frac!(2)));
        let mut aug2 = aug1.clone_owned();

        gauss_eliminate(&mut aug2);
        println!("{}", aug1);
        println!("{}", aug2);
        assert!(aug2.fixed_view::<2, 2>(1, 0).lower_triangle().is_zero());
        assert!(aug2.fixed_view::<2, 2>(0, 1).upper_triangle().is_zero());
    }
}
