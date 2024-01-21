//! A symmetry operation in 3D space, considered geometrically. Acts on 3D space as an [`Isometry`],
//! but specifically considers the subgroup of isometries that occurs in crystal space groups.

use std::f64::consts::TAU;
use std::iter::successors;
use std::ops::SubAssign;

use nalgebra::{
    matrix, vector, Const, Matrix3, Matrix4, OMatrix, Point3, Rotation3, SMatrix, Translation3,
    Unit, Vector3,
};
use num_traits::Zero;
use simba::scalar::{SubsetOf, SupersetOf};
use thiserror::Error;

use crate::frac;
use crate::{
    frac::{BaseInt, Frac, DENOM},
    isometry::Isometry,
};

/// The axis of rotation. Distinguishes a single point on the line, which is the center of a
/// potential rotoinversion.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct RotationAxis {
    origin: Point3<Frac>,
    v: Vector3<Frac>,
}

/// Reduces coefs by a common denominator and scales so the first nonzero coordinate is positive.
fn reduce_coefs(v: &Vector3<Frac>) -> (Vector3<Frac>, i16) {
    let nonzero_elems: Vec<BaseInt> = v
        .iter()
        .map(|e| if e.is_zero() { None } else { Some(e.numerator) })
        .flatten()
        .collect();
    let scaling_factor = match nonzero_elems[..] {
        [] => 0,
        [i1] => i1,
        [i1, i2] => Frac::gcd(i1.abs(), i2.abs()) * i1.signum(),
        [i1, i2, i3] => Frac::gcd(Frac::gcd(i1.abs(), i2.abs()), i3.abs()) * i1.signum(),
        _ => panic!("Should be unreachable"),
    };

    (v.map(|e| e / frac!(scaling_factor)), scaling_factor)
}

impl RotationAxis {
    /// Vector oriented as v going through origin.
    pub fn new(v: Vector3<Frac>, origin: Point3<Frac>) -> Self {
        Self {
            origin,
            v: reduce_coefs(&v).0,
        }
    }

    /// Return representation as origin and vector.
    pub fn as_origin_vector(&self) -> (Point3<Frac>, Vector3<Frac>) {
        (self.origin, self.v)
    }
}

/// The kind of rotation: sense and order.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum RotationKind {
    Two,
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
            (_, 2) => Self::Two,
            (true, 3) => Self::PosThree,
            (false, 3) => Self::NegThree,
            (true, 4) => Self::PosFour,
            (false, 4) => Self::NegFour,
            (true, 6) => Self::PosSix,
            (false, 6) => Self::NegSix,
            _ => panic!("Invalid sense and order: {}, {}", is_ccw, order),
        }
    }

    pub fn as_frac(&self) -> Frac {
        match &self {
            RotationKind::Two => frac!(1 / 2),
            RotationKind::PosThree => frac!(1 / 3),
            RotationKind::PosFour => frac!(1 / 4),
            RotationKind::PosSix => frac!(1 / 6),
            RotationKind::NegThree => frac!(-1 / 3),
            RotationKind::NegFour => frac!(-1 / 4),
            RotationKind::NegSix => frac!(-1 / 6),
        }
    }
}

/// The amount of translation along the axis of rotation in a screw rotation.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct ScrewOrder {
    order: BaseInt,
}

impl ScrewOrder {
    pub fn new_order(order: BaseInt) -> Self {
        Self { order }
    }
}

/// A plane (particularly of reflection.)
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct Plane {
    n: Vector3<Frac>,
    d: Frac,
}

impl Plane {
    pub fn from_normal_and_d(n: Vector3<Frac>, d: Frac) -> Self {
        let (_n_scaled, scale) = reduce_coefs(&n);
        let d_scale = (Frac::gcd(scale.abs(), d.numerator.abs()) * scale.signum()).into();
        Self {
            n: n.map(|e| e / d_scale),
            d: d / d_scale,
        }
    }
    pub fn from_basis_and_origin(
        v1: Vector3<Frac>,
        v2: Vector3<Frac>,
        origin: Point3<Frac>,
    ) -> Self {
        let normal = v1.cross(&v2);
        let d = -normal.dot(&origin.coords);
        Self::from_normal_and_d(normal, d)
    }

    pub fn reflection_matrix(&self) -> Matrix4<f64> {
        // for now, use f64

        // https://www.wikiwand.com/en/Transformation_matrix#Reflection_2
        let fl_n: Vector3<f64> = self.n.to_subset_unchecked();
        let fl_n_norm = fl_n.norm();
        let &[a, b, c] = fl_n.scale(1. / fl_n_norm).as_slice() else {
            panic!()
        };
        let d: f64 = self.d.to_subset_unchecked();
        let d = d / fl_n_norm;
        let abcd = [a, b, c, d];
        Matrix4::from_fn(|i, j| {
            let entry = if i == j { 1. } else { 0. };
            if i == 3 {
                entry
            } else {
                entry - 2. * abcd[i] * abcd[j]
            }
        })
    }
}

/// A symmetry operation. See section 1.2.1 of the International Tables of Crystallography.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SymmOp {
    /// The identity.
    Identity,
    /// Inversion around a point.
    Inversion(Point3<Frac>),
    /// A translation.
    Translation(Vector3<Frac>),
    /// A rotation.
    Rotation(RotationAxis, RotationKind),
    /// A rotoinversion. Rotoinversions of order 2 are reflections and should be treated as such.
    Rotoinversion(RotationAxis, RotationKind),
    /// A screw rotation or rotoinversion. Screw rotoinversions of order 2 are glide reflections.
    Screw(RotationAxis, RotationKind, bool, ScrewOrder, Vector3<Frac>),
    /// A reflection through a plane.
    Reflection(Plane),
    /// A glide reflection.
    Glide(Plane, Vector3<Frac>),
}

impl SymmOp {
    pub fn new_generalized_rotation(
        axis: RotationAxis,
        kind: RotationKind,
        is_proper: bool,
        tau: Vector3<Frac>,
    ) -> Self {
        match (is_proper, tau.is_zero()) {
            (true, true) => Self::Rotation(axis, kind),
            (false, true) => Self::Rotoinversion(axis, kind),
            (proper, false) => Self::Screw(axis, kind, proper, ScrewOrder { order: 1 }, tau),
        }
    }

    pub fn new_generalized_reflection(plane: Plane, tau: Vector3<Frac>) -> Self {
        if tau.is_zero() {
            Self::Reflection(plane)
        } else {
            Self::Glide(plane, tau)
        }
    }

    /// Gets the isometry corresponding to the geometric operation.
    pub fn to_iso(&self) -> Isometry {
        match &self {
            SymmOp::Identity => Isometry::identity(),
            SymmOp::Inversion(tau) => Isometry::new_rot_tau(
                Matrix3::identity().scale(frac!(-1)),
                matrix![
                    tau.x;
                    tau.y;
                    tau.z;
                ],
            ),
            SymmOp::Translation(tau) => {
                Isometry::new_rot_tau(Matrix3::identity(), tau.clone_owned())
            }
            SymmOp::Rotation(axis, kind)
            | SymmOp::Rotoinversion(axis, kind)
            | SymmOp::Screw(axis, kind, _, _, _) => {
                // I am going to 100% punt on doing this without numerical instability.
                let th = f64::from(kind.as_frac()) * TAU;
                let (o, u) = axis.as_origin_vector();
                let o: Point3<f64> = o.to_subset_unchecked();
                let u: Vector3<f64> = u.to_subset_unchecked();
                let u = Unit::new_normalize(u);
                let rot = Matrix4::from_axis_angle(&u, th);
                let inv_mat: Matrix4<f64> = Matrix4::from_partial_diagonal(&[-1., -1., -1., 1.]);
                let trans = Translation3::new(o.x, o.y, o.z).to_homogeneous();
                let trans_inv = Translation3::new(-o.x, -o.y, -o.z).to_homogeneous();
                let affine = match self {
                    SymmOp::Rotation(_, _) => trans * rot * trans_inv,
                    SymmOp::Rotoinversion(_, _) => trans * inv_mat * rot * trans_inv,
                    SymmOp::Screw(_, _, is_proper, _order, tau) => {
                        let roto_inv = if *is_proper { rot } else { inv_mat * rot };
                        let tau_m: Vector3<f64> = tau.to_subset_unchecked();
                        Translation3::new(tau_m.x, tau_m.y, tau_m.z).to_homogeneous()
                            * trans
                            * roto_inv
                            * trans_inv
                    }
                    _ => panic!(),
                };

                // moment of truth...
                let affine_frac = Matrix4::from_iterator(
                    affine.into_iter().map(|&fl| Frac::from_f64_unchecked(fl)),
                );
                affine_frac.try_into().unwrap()
            }
            SymmOp::Reflection(plane) => {
                // moment of truth...
                let affine_frac = Matrix4::from_iterator(
                    plane
                        .reflection_matrix()
                        .into_iter()
                        .map(|&fl| Frac::from_f64_unchecked(fl)),
                );
                affine_frac.try_into().unwrap()
            }
            SymmOp::Glide(plane, tau) => {
                // moment of truth...
                let tau_f: Vector3<f64> = tau.to_subset_unchecked();
                let tau_m = Translation3::new(tau_f.x, tau_f.y, tau_f.z).to_homogeneous();
                let affine = tau_m * plane.reflection_matrix();
                // println!(
                //     "Plane {:?}: {}",
                //     plane,
                //     plane
                //         .reflection_matrix()
                //         .map(|f| (f * 1000.).round() / 1000.)
                // );
                let affine_frac = Matrix4::from_iterator(
                    affine.into_iter().map(|&fl| Frac::from_f64_unchecked(fl)),
                );
                affine_frac.try_into().unwrap()
            }
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum SymmOpError {
    #[error("Matrix not symmetry operation: {0:?}")]
    NotSymmetryOperation(Isometry),
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
        let W = m.rot();
        let w = m.tau();
        let det = det3x3(W);
        let is_proper = if det == Frac::ONE {
            true
        } else if det == Frac::NEG_ONE {
            false
        } else {
            return err;
        };
        // (a) (ii) Classify angle according to trace
        let tr = W.diagonal().sum();
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

        // println!("tr:{}\ndet:{}\ntype:{}\norder:{}", tr, det, rot_type, order);

        if rot_type == 1 {
            // translation or identity, no rotational component
            if w == Vector3::<Frac>::zeros() {
                return Ok(Self::Identity);
            } else {
                return Ok(Self::Translation(w));
            }
        } else if rot_type == -1 {
            // inversion, find center
            // solve for fixed point to find center
            // -p + τ = p
            // p = τ/2
            return Ok(Self::Inversion(w.scale(Frac::ONE_HALF).into()));
        } else {
            // (b) find rotation axis

            // Consider Y(W) = W^(k-1) + W^(k-2) + ... + W + I where k is the order of the rotation.
            // (Just the rotation.) Note that W Y(W) = I + W^(k-1) + ... + W^2 + W = Y(W), so Y(W)
            // sends anything not orthogonal to it to the axis. So we can try Y(W) v and optionally
            // try a second v if we get unlucky and Y(W) v = 0. If we have a rotoinversion, then
            // instead of Y(W) we have Y(-W).
            let k = rot_type.abs();
            let W_abs = W * det;
            let I: Matrix3<Frac> = Matrix3::identity();
            let Y: Matrix3<Frac> = successors(Some(I), |acc| Some(W_abs * acc))
                .take(k as usize)
                .sum();

            // Y(W) is invariant, so any vector that doesn't go to 0 will work. If we pick the
            // vectors <100>, <010>, and <001> to try, it's basically equivalent to checking each
            // column and picking the first one that's nonzero.
            assert!(!Y.is_zero());

            let axis = (0..3)
                .map(|c| {
                    if Y.column(c).abs().sum().is_zero() {
                        None
                    } else {
                        Some(Y.column(c))
                    }
                })
                .flatten()
                .next()
                .unwrap()
                .clone_owned();

            let rot_kind = if order > 2 {
                // (c) sense of rotation

                // the sense is, where u is the axis, x is some vector not parallel to u, and d = det W

                // | u1 x1 (dWx)1 |
                // | u2 x2 (dWx)2 |
                // | u3 x3 (dWx)3 |

                // We can pick x to be one of the basis vectors.
                let sense: Frac = [Vector3::x(), Vector3::y(), Vector3::z()]
                    .into_iter()
                    .map(|x| {
                        let Z = Matrix3::from_columns(&[axis, x, W_abs * x]);
                        let z_det = det3x3(Z);
                        if z_det.is_zero() {
                            None
                        } else {
                            Some(z_det)
                        }
                    })
                    .flatten()
                    .next()
                    .expect(format!("{} {}", axis, W).as_str());

                // todo this det is not indicated, but it's the only way I get it to work?
                let sense = sense * det;

                // println!("Y(±W): {}", Y);
                // println!(
                //     "Z: {}",
                //     matrix![
                //         axis.x, frac!(0), W_abs.m13;
                //         axis.y, frac!(0), W_abs.m23;
                //         axis.z, frac!(1), W_abs.m33
                //     ]
                // );
                // println!("W: {}\naxis: {}\ndet: {}\nsense: {}", W, axis, det, sense);

                let sense = sense.numerator.signum();
                // this would mean our x logic above failed
                assert!(sense != 0 || order == 2, "{} is 0???", sense);
                let is_ccw = sense == 1;
                Some(RotationKind::new(is_ccw, rot_type.abs() as usize))
            } else {
                // reflection
                None
            };

            // (2) Analysis of the translation column τ, which they call w

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
                        RotationAxis::new(axis, center),
                        rot_kind.unwrap(),
                        is_proper,
                        w_g,
                    ))
                }
                [_u] => {
                    // single axis: rotation or screw
                    Ok(Self::new_generalized_rotation(
                        RotationAxis::new(axis, center),
                        rot_kind.unwrap(),
                        is_proper,
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
        // example 3 from ITA 1.2.2.4
        let iso = Isometry::from_str("-y + 3/4, -x + 1/4, z + 1/4").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        let ans = SymmOp::Glide(
            Plane::from_basis_and_origin(
                Vector3::new(frac!(1), frac!(-1), frac!(0)),
                Vector3::new(frac!(0), frac!(0), frac!(1)),
                Point3::new(frac!(0), frac!(-1 / 2), frac!(-1 / 2)),
            ),
            Vector3::new(frac!(1 / 4), frac!(-1 / 4), frac!(1 / 4)),
        );

        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(),
            ans.to_iso(),
            "{}\n{}",
            symm.to_iso().mat(),
            ans.to_iso().mat()
        )
    }

    #[test]
    fn test_rotation() {
        // example 1 from ITA 1.2.2.4
        let iso = Isometry::from_str("y+1/4, -x+1/4, z+3/4").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        let ans = SymmOp::new_generalized_rotation(
            RotationAxis::new(Vector3::z(), Point3::new(frac!(-1 / 4), frac!(0), frac!(0))),
            RotationKind::NegFour,
            true,
            Vector3::new(frac!(0), frac!(0), frac!(3 / 4)),
        );

        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(),
            ans.to_iso(),
            "{}\n{}",
            symm.to_iso().mat(),
            ans.to_iso().mat()
        )
    }

    #[test]
    fn test_rotoinversion() {
        // example 2 from ITA 1.2.2.4
        let iso = Isometry::from_str("-z+1/2, x+1/2, y").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        let ans = SymmOp::new_generalized_rotation(
            RotationAxis::new(
                Vector3::new(frac!(-1), frac!(1), frac!(-1)),
                Point3::new(frac!(0), frac!(-1 / 2), frac!(-1 / 2)),
            ),
            RotationKind::PosThree,
            false,
            Vector3::zero(),
        );
        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(),
            ans.to_iso(),
            "{}\n{}",
            symm.to_iso().mat(),
            ans.to_iso().mat()
        )
    }

    #[test]
    fn test_row_reduce() {
        let mut aug1: SMatrix<Frac, 3, 7> = SMatrix::identity();
        aug1.set_column(0, &Vector3::new(frac!(2), frac!(-3), frac!(-2)));
        aug1.set_column(1, &Vector3::new(frac!(1), frac!(-1), frac!(1)));
        aug1.set_column(2, &Vector3::new(frac!(-1), frac!(2), frac!(2)));
        let mut aug2 = aug1.clone_owned();

        gauss_eliminate(&mut aug2);
        // println!("{}", aug1);
        // println!("{}", aug2);
        assert!(aug2.fixed_view::<2, 2>(1, 0).lower_triangle().is_zero());
        assert!(aug2.fixed_view::<2, 2>(0, 1).upper_triangle().is_zero());
    }
}
