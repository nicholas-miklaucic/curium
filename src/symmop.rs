//! A symmetry operation in 3D space, considered geometrically. Acts on 3D space as an [`Isometry`],
//! but specifically considers the subgroup of isometries that occurs in crystal space groups.

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::f64::consts::TAU;
use std::fmt::Display;
use std::iter::successors;
use std::ops::Mul;

use nalgebra::{
    matrix, ComplexField, Const, Matrix3, Matrix4, OMatrix, Point, Point3, SMatrix, Translation3,
    Unit, Vector3,
};
use num_traits::{FromPrimitive, Signed, Zero};
use simba::scalar::SupersetOf;
use thiserror::Error;

use crate::algebra::GroupElement;
use crate::fract::FracError;
use crate::geometry::{Direction, Plane, RotationAxis, SymmetryElement};
use crate::hermann_mauguin::PartialSymmOp;
use crate::markup::{Block, RenderBlocks, ITA, UNICODE};
use crate::symbols::{LPAREN, SPACE, SUP_MINUS, SUP_PLUS, TAB};
use crate::{
    frac,
    fract::{BaseInt, Frac, DENOM},
    isometry::Isometry,
};

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

    /// The inverse direction, switching `Pos` with `Neg`.
    pub fn inv(&self) -> Self {
        match &self {
            Self::Two => Self::Two,
            Self::PosThree => Self::NegThree,
            Self::PosFour => Self::NegFour,
            Self::PosSix => Self::NegSix,
            Self::NegThree => Self::PosThree,
            Self::NegFour => Self::PosFour,
            Self::NegSix => Self::PosSix,
        }
    }

    /// The order of the rotation.
    pub fn order(&self) -> BaseInt {
        match &self {
            Self::Two => 2,
            Self::PosThree | Self::NegThree => 3,
            Self::PosFour | Self::NegFour => 4,
            Self::PosSix | Self::NegSix => 6,
        }
    }
}

/// A proper rotation around an axis. This should probably not be worked with directly: it's mainly
/// to reduce code duplication between `SymmOp::Rotation`, `SymmOp::Rotoinversion`, and
/// `SymmOp::Screw`.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct SimpleRotation {
    pub axis: RotationAxis,
    pub kind: RotationKind,
}

impl SimpleRotation {
    /// New rotation from axis and kind.
    pub fn new(axis: RotationAxis, kind: RotationKind) -> Self {
        Self { axis, kind }
    }

    /// Flips the axis and kind, which preserves the meaning of the rotation.
    pub fn as_opposite_axis(&self) -> Self {
        Self::new(self.axis.inv(), self.kind.inv())
    }

    /// Reduces to modulo a unit cell.
    pub fn modulo_unit_cell(&self) -> Self {
        Self {
            axis: self.axis.modulo_unit_cell(),
            kind: self.kind,
        }
    }

    /// Gets the rotation matrix, with optional inversion.
    pub fn rot_matrix(&self, do_inv: bool, is_hex: bool) -> Matrix4<Frac> {
        let origin = self.axis.conventional().origin();
        let &[v1, v2, v3] = self
            .axis
            .dir()
            .conventional_orientation()
            .as_vec3()
            .as_slice()
        else {
            panic!()
        };

        let (v1, v2, v3) = (
            v1.numerator / Frac::DENOM,
            v2.numerator / Frac::DENOM,
            v3.numerator / Frac::DENOM,
        );

        let f0 = frac!(0);
        let f1 = frac!(1);
        let fm1 = frac!(-1);

        // dbg!((v1, v2, v3, self.kind.order()));
        let m = match (v1, v2, v3, self.kind.order()) {
            // https://cci.lbl.gov/sginfo/hall_symbols.html
            (_, _, _, 1) => Matrix3::identity(),
            (1, 0, 0, 2) if is_hex => matrix![
                f1, fm1, f0;
                f0, fm1, f0;
                f0, f0, fm1;
            ],
            (1, 0, 0, 2) if !is_hex => matrix![
                f1, f0, f0;
                f0, fm1, f0;
                f0, f0, fm1;
            ],
            (0, 1, 0, 2) if !is_hex => matrix![
                fm1, f0, f0;
                f0, f1, f0;
                f0, f0, fm1;
            ],
            (0, 1, 0, 2) if is_hex => matrix![
                fm1, f0, f0;
                fm1, f1, f0;
                f0, f0, fm1;
            ],
            (0, 0, 1, 2) => matrix![
                fm1, f0, f0;
                f0, fm1, f0;
                f0, f0, f1;
            ],
            (1, 0, 0, 3) => matrix![
                f1, f0, f0;
                f0, f0, fm1;
                f0, f1, fm1;
            ],
            (0, 1, 0, 3) => matrix![
                fm1, f0, f1;
                f0, f1, f0;
                fm1, f0, f0;
            ],
            (0, 0, 1, 3) => matrix![
                f0, fm1, f0;
                f1, fm1, f0;
                f0, f0, f1;
            ],
            (1, 0, 0, 4) => matrix![
                f1, f0, f0;
                f0, f0, fm1;
                f0, f1, f0;
            ],
            (0, 1, 0, 4) => matrix![
                f0, f0, f1;
                f0, f1, f0;
                fm1, f0, f0;
            ],
            (0, 0, 1, 4) => matrix![
                f0, fm1, f0;
                f1, f0, f0;
                f0, f0, f1;
            ],
            (1, 0, 0, 6) => matrix![
                f1, f0, f0;
                f0, f1, fm1;
                f0, f1, f0;
            ],
            (0, 1, 0, 6) => matrix![
                f0, f0, f1;
                f0, f1, f0;
                fm1, f0, f1;
            ],
            (0, 0, 1, 6) => matrix![
                f1, fm1, f0;
                f1, f0, f0;
                f0, f0, f1;
            ],
            (0, 1, -1, 2) => matrix![
                fm1, f0, f0;
                f0, f0, fm1;
                f0, fm1, f0;
            ],
            (0, 1, 1, 2) => matrix![
                fm1, f0, f0;
                f0, f0, f1;
                f0, f1, f0;
            ],
            (1, 0, -1, 2) => matrix![
                f0, f0, fm1;
                f0, fm1, f0;
                fm1, f0, f0;
            ],
            (1, 0, 1, 2) => matrix![
                f0, f0, f1;
                f0, fm1, f0;
                f1, f0, f0;
            ],
            (1, -1, 0, 2) => matrix![
                f0, fm1, f0;
                fm1, f0, f0;
                f0, f0, fm1;
            ],
            (1, 1, 0, 2) => matrix![
                f0, f1, f0;
                f1, f0, f0;
                f0, f0, fm1;
            ],
            (1, 2, 0, 2) => matrix![
                fm1, f1, f0;
                f0, f1, f0;
                f0, f0, fm1;
            ],
            (2, 1, 0, 2) => matrix![
                f1, f0, f0;
                f1, fm1, f0;
                f0, f0, fm1;
            ],
            (1, -2, 0, 2) => matrix![
                fm1, f0, f0;
                fm1, f1, f0;
                f0, f0, fm1;
            ],
            (2, -1, 0, 2) => matrix![
                f1, fm1, f0;
                f0, fm1, f0;
                f0, f0, fm1;
            ],
            (1, 1, 1, 3) => matrix![
                f0, f0, f1;
                f1, f0, f0;
                f0, f1, f0;
            ],
            (1, -1, -1, 3) => matrix![
                f0, f0, fm1;
                fm1, f0, f0;
                f0, f1, f0;
            ],
            (-1, 1, -1, 3) => matrix![
                f0, f0, f1;
                fm1, f0, f0;
                f0, fm1, f0;
            ],
            (-1, -1, 1, 3) => matrix![
                f0, f0, fm1;
                f1, f0, f0;
                f0, fm1, f0;
            ],
            (a, b, c, d) => {
                dbg!(a, b, c, d);
                panic!("Uh");
            }
        };

        // we can do inverse, for negative sense, by simply applying order - 1 times
        let m = if self.kind.as_frac().is_negative() {
            let mut new_m = Matrix3::<Frac>::identity();
            for _ in 0..(self.kind.order() - 1) {
                new_m *= m;
            }
            new_m
        } else {
            m
        };

        let m = m.to_homogeneous();

        let o = origin;

        let tau = Translation3::new(o.x, o.y, o.z);
        // apply origin shift
        let shift = tau.to_homogeneous();
        let shift_inv = tau.inverse().to_homogeneous();

        let inv_mat = matrix![
            fm1, f0, f0, f0;
            f0, fm1, f0, f0;
            f0, f0, fm1, f0;
            f0, f0, f0, f1;
        ];
        if do_inv {
            let mut v = origin.coords.to_homogeneous();
            v[(3, 0)] = f1;
            // println!(
            //     "--{}\n{}\n{}\n{}\n{}",
            //     v,
            //     shift_inv * v,
            //     m * shift_inv * v,
            //     inv_mat * m * shift_inv * v,
            //     shift * inv_mat * m * shift_inv * v
            // );
            shift * inv_mat * m * shift_inv
        } else {
            shift * m * shift_inv
        }
    }
}

/// The order of a screw rotation, defined as such: Take the order of the screw operation, up to a
/// unit cell equivalence. (The order is infinite otherwise.) The screw order is the number of
/// copies of the direction vector that the point has moved. For example, for a 3-fold rotation, an
/// order of 1 indicates that the screw is 1/3 of the distance between the point and its symmetric
/// equivalent in the neighboring unit cell.
///
/// In ITA, they generally only consider screws up to translation, so instead of having screws of
/// negative order they wrap around: a -1 screw in a 6-fold rotation is equivalent to a 5 screw up
/// to a unit cell. [`SymmOp`] does not assume equivalence up to a unit cell, so negative screw
/// orders are necessary.
pub type ScrewOrder = Frac;

/// Whether a rotation is `Proper` (determinant 1, preserves orientation) or `Improper` (determinant
/// -1, flips orientation).
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum RotationDirectness {
    /// A proper rotation.
    Proper,
    /// An improper rotation, also called a rotoreflection or rotoinversion.
    Improper,
}

impl RotationDirectness {
    /// The sign of the determimnant: -1 or 1.
    pub fn det_sign(&self) -> BaseInt {
        match &self {
            RotationDirectness::Proper => 1,
            RotationDirectness::Improper => -1,
        }
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
    Rotation(SimpleRotation),
    /// A rotoinversion. Rotoinversions of order 2 are reflections and should be treated as such.
    Rotoinversion(SimpleRotation),
    /// A screw rotation or rotoinversion. Screw rotoinversions of order 2 are glide reflections.
    Screw(SimpleRotation, RotationDirectness, ScrewOrder),
    /// A reflection through a plane.
    Reflection(Plane),
    /// A glide reflection.
    Glide(Plane, Vector3<Frac>),
}

impl RenderBlocks for SymmOp {
    fn components(&self) -> Vec<Block> {
        let partial = PartialSymmOp::try_from_op(self);
        let mut blocks = match (*self, partial) {
            (SymmOp::Inversion(tau), _) => {
                let mut compound = vec![Block::new_int(-1)];
                compound.extend(tau.components());
                compound
            }
            (SymmOp::Translation(tau), _) => vec![
                Block::new_text("t"),
                Block::Point3D(
                    Block::Blocks(tau[0].components()).into(),
                    Block::Blocks(tau[1].components()).into(),
                    Block::Blocks(tau[2].components()).into(),
                ),
            ],
            (SymmOp::Glide(_plane, tau), None) => vec![
                Block::new_text("g"),
                Block::Point3D(
                    Block::Blocks(tau[0].components()).into(),
                    Block::Blocks(tau[1].components()).into(),
                    Block::Blocks(tau[2].components()).into(),
                ),
            ],

            (_, Some(partial)) => partial.components(),
            _ => {
                dbg!(partial, self);
                panic!("Oops");
            }
        };

        // add +/- sign
        if let Some(rot) = self.rotation_component() {
            let kind = rot.kind;
            if kind.order() > 2 {
                if kind.as_frac().is_positive() {
                    blocks.push(SUP_PLUS);
                } else {
                    blocks.push(SUP_MINUS);
                }
            }
        }

        blocks.push(SPACE);
        blocks.push(TAB);
        blocks.extend(self.symmetry_element().components());

        // rotoinversions need an additional indication of the center
        if let (Some(o), Some(_)) = (self.inversion_component(), self.rotation_component()) {
            blocks.push(Block::new_text(": "));
            blocks.extend(o.components());
        }

        blocks
    }
}

impl RenderBlocks for Point3<Frac> {
    fn components(&self) -> Vec<Block> {
        vec![Block::Point3D(
            Block::Blocks(self.x.components()).into(),
            Block::Blocks(self.y.components()).into(),
            Block::Blocks(self.z.components()).into(),
        )]
    }
}

impl std::fmt::Display for SymmOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", UNICODE.render_to_string(self))
    }
}

impl SymmOp {
    /// Creates a `SymmOp` from a rotation that may or may not be a rotoinversion and may or may not
    /// have a screw translation.
    pub fn new_generalized_rotation(
        axis: RotationAxis,
        kind: RotationKind,
        is_proper: bool,
        tau: Vector3<Frac>,
    ) -> Self {
        let rot = SimpleRotation::new(axis, kind);
        let rot_op = match (is_proper, tau.is_zero()) {
            (true, true) => Self::Rotation(rot),
            (false, true) => Self::Rotoinversion(rot),
            (proper, false) => {
                let screw_order = axis.dir().compute_scale(tau).unwrap_or_else(|| {
                    panic!("Bad screw translation {} for dir {}", tau, axis.dir())
                });
                // let screw_order = screw_scale * Frac::from(kind.order());
                // let screw_order = if screw_order.numerator % Frac::DENOM != 0 {
                //     dbg!(screw_order, axis, tau, screw_scale, kind, is_proper);
                //     panic!("Screw translation not proper scale");
                // } else {
                //     screw_order.numerator / Frac::DENOM
                // };
                Self::Screw(
                    rot,
                    if proper {
                        RotationDirectness::Proper
                    } else {
                        RotationDirectness::Improper
                    },
                    screw_order,
                )
            }
        };

        rot_op.conventional()
    }

    pub fn new_generalized_reflection(plane: Plane, tau: Vector3<Frac>) -> Self {
        let reflect = if tau.is_zero() {
            Self::Reflection(plane)
        } else {
            Self::Glide(plane, tau)
        };
        reflect.conventional()
    }

    /// Returns a normalized version, changing the representation of the SymmOp to comply with ITA
    /// conventions. The actual meaning and the associated isometry do not change.
    ///
    /// An example of when normalization is necessary: positive rotation around the axis \[1 -1 1\]
    /// and negative rotation around the axis \[-1 1 -1\] are equivalent. ITA uses the latter
    /// representation, for reasons that are unclear to me, and so when comparing `SymmOp` values to
    /// other references it's important to flip this around. In other cases, it makes a lot more
    /// sense: the axis \[001\] is much more natural than its flipped equivalent.
    pub fn conventional(&self) -> Self {
        // for some reason, the convention seems to be an even number of negative signs in the axis.
        // I have no idea why this is, and it's not explained at all in ITA, at least where I can
        // find. It might be some deep result of how they generate the symmetry directions for the
        // space group.
        match self {
            SymmOp::Identity => *self,
            SymmOp::Inversion(_) => *self,
            SymmOp::Translation(_) => *self,
            SymmOp::Rotation(rot) | SymmOp::Rotoinversion(rot) | SymmOp::Screw(rot, _, _) => {
                if !rot.axis.is_conventionally_oriented() {
                    match self {
                        SymmOp::Rotation(rot) => SymmOp::Rotation(rot.as_opposite_axis()),
                        SymmOp::Rotoinversion(rot) => SymmOp::Rotoinversion(rot.as_opposite_axis()),
                        SymmOp::Screw(rot, directness, order) => {
                            SymmOp::Screw(rot.as_opposite_axis(), *directness, -*order)
                        }
                        _ => panic!(),
                    }
                } else {
                    *self
                }
            }
            SymmOp::Reflection(pl) => SymmOp::Reflection(pl.conventional()),
            SymmOp::Glide(pl, tau) => SymmOp::Glide(pl.conventional(), *tau),
        }
    }

    /// Reduces the `SymmOp` to equivalence up to a unit cell. Note that this does *not* mean that
    /// the `SymmOp` keeps its inputs inside their unit cell, just that the symmetry elements and
    /// translations are reduced. For example, a translation by <1/2, 0, 0> will still map (3/4, 0,
    /// 0) to (5/4, 0, 0) and will be unchanged, but a translation by <11/2, 0, 0> will be mapped to
    /// <1/2, 0, 0>.
    pub fn modulo_unit_cell_by_elements(&self) -> Self {
        fn mod_unit(f: Frac) -> Frac {
            f.modulo_one()
        }
        match *self {
            SymmOp::Identity => SymmOp::Identity,
            SymmOp::Inversion(tau) => SymmOp::Inversion(tau.map(mod_unit)),
            SymmOp::Translation(tau) => SymmOp::Translation(tau.map(mod_unit)),
            SymmOp::Rotation(rot) => SymmOp::Rotation(rot.modulo_unit_cell()),
            SymmOp::Rotoinversion(rot) => SymmOp::Rotoinversion(rot.modulo_unit_cell()),
            SymmOp::Screw(rot, dir, tau) => {
                SymmOp::Screw(rot.modulo_unit_cell(), dir, tau.modulo_one())
            }
            SymmOp::Reflection(plane) => SymmOp::Reflection(plane.modulo_unit_cell()),
            SymmOp::Glide(plane, tau) => SymmOp::Glide(plane.modulo_unit_cell(), tau.map(mod_unit)),
        }
    }

    /// Reduces the `SymmOp` to equivalence up to a unit cell. Note that this does *not* mean that
    /// the `SymmOp` keeps its inputs inside their unit cell, just that the symmetry elements and
    /// translations are reduced. For example, a translation by <1/2, 0, 0> will still map (3/4, 0,
    /// 0) to (5/4, 0, 0) and will be unchanged, but a translation by <11/2, 0, 0> will be mapped to
    /// <1/2, 0, 0>.
    pub fn modulo_unit_cell(&self) -> Self {
        Self::classify_affine(self.to_iso(false).modulo_unit_cell()).unwrap()
    }

    /// Gets the rotation associated with the operation, if one exists, otherwise `None`.
    pub fn rotation_component(&self) -> Option<SimpleRotation> {
        match *self {
            Self::Rotation(rot) => Some(rot),
            Self::Rotoinversion(rot) => Some(rot),
            Self::Screw(rot, _dir, _tau) => Some(rot),
            _ => None,
        }
    }

    /// Gets the center associated with the inversion component, if any.
    pub fn inversion_component(&self) -> Option<Point3<Frac>> {
        match *self {
            SymmOp::Inversion(o) => Some(o),
            SymmOp::Rotoinversion(rot) | SymmOp::Screw(rot, RotationDirectness::Improper, _) => {
                Some(rot.axis.origin())
            }
            _ => None,
        }
    }

    /// Gets the mirror plane associated with the operation, if one exists, otherwise `None`.
    pub fn reflection_component(&self) -> Option<Plane> {
        match *self {
            Self::Reflection(plane) => Some(plane),
            Self::Glide(plane, _tau) => Some(plane),
            _ => None,
        }
    }

    /// Gets the translation component of the operation. Note that this is *not* the same as the
    /// translation component of the underlying affine transformation. An inversion through the
    /// point (1/4, 0, 0) is not a linear transformation, as moves the origin to (1/2, 0, 0). But as
    /// defined here its translation is `None`, because after the inversion no additional
    /// translation is needed.
    pub fn translation_component(&self) -> Option<Vector3<Frac>> {
        match *self {
            Self::Translation(tau) => Some(tau),
            Self::Screw(rot, _dir, tau) => Some(rot.axis.dir().scaled_vec(tau)),
            Self::Glide(_plane, tau) => Some(tau),
            _ => None,
        }
    }

    /// Gets the symmetry direction of an operation:
    /// - For rotations, rotoinversions, and screws, this is the direction of the rotation axis.
    /// - For reflections and glide reflections, this is the direction of the normal vector of the
    ///   plane.
    /// - For the identity, inversion, and translations, there is no symmetry direction and `None`
    ///   is returned.
    pub fn symmetry_direction(&self) -> Option<Direction> {
        self.rotation_component()
            .map(|rot| rot.axis.dir())
            .or_else(|| self.reflection_component().map(|pl| pl.normal()))
    }

    /// Gets the *symmetry element* of the operation. This is defined as the locus of points left
    /// unchanged by the linear part of the operation.
    /// - For pure inversions, this is the rotation center.
    /// - For rotations of any kind, this is the rotation axis.
    /// - For reflections and glides, this is the plane.
    /// - For the identity and translations, this is the whole space.
    pub fn symmetry_element(&self) -> SymmetryElement {
        match *self {
            SymmOp::Identity | SymmOp::Translation(_) => SymmetryElement::Space,
            SymmOp::Inversion(o) => SymmetryElement::Point(o),
            SymmOp::Rotation(rot) | SymmOp::Rotoinversion(rot) | SymmOp::Screw(rot, _, _) => {
                SymmetryElement::Line(rot.axis)
            }
            SymmOp::Reflection(plane) | SymmOp::Glide(plane, _) => SymmetryElement::Plane(plane),
        }
    }

    /// Gets the isometry corresponding to the geometric operation.
    pub fn to_iso(&self, is_hex: bool) -> Isometry {
        match &self {
            SymmOp::Identity => Isometry::identity(),
            SymmOp::Inversion(tau) => Isometry::new_rot_tau(
                Matrix3::identity().scale(frac!(-1)),
                tau.coords.scale(frac!(2)),
            ),
            SymmOp::Translation(tau) => {
                Isometry::new_rot_tau(Matrix3::identity(), tau.clone_owned())
            }
            SymmOp::Rotation(rot) => Isometry::new_affine(rot.rot_matrix(false, is_hex)),
            SymmOp::Rotoinversion(rot) => Isometry::new_affine(rot.rot_matrix(true, is_hex)),
            SymmOp::Screw(rot, dir, screw) => {
                let tau_m = Translation3::from(rot.axis.dir().scaled_vec(*screw)).to_homogeneous();

                Isometry::new_affine(
                    tau_m * rot.rot_matrix(dir == &RotationDirectness::Improper, is_hex),
                )
            }
            SymmOp::Reflection(plane) => Isometry::new_affine(
                SimpleRotation::new(plane.normal_axis(), RotationKind::Two)
                    .rot_matrix(true, is_hex),
            ),
            SymmOp::Glide(plane, tau) => {
                let tau_m = Translation3::new(tau.x, tau.y, tau.z).to_homogeneous();

                let ref_m = SymmOp::Reflection(*plane).to_iso(is_hex).mat();

                Isometry::new_affine(tau_m * ref_m)
            }
        }
    }

    /// Decomposes the symmetry operation into a rotation and translation component.
    pub fn to_rot_and_tau(&self) -> (SymmOp, Vector3<Frac>) {
        match self {
            SymmOp::Identity => (SymmOp::Identity, Vector3::zero()),
            SymmOp::Inversion(tau) => (
                SymmOp::Inversion(Point3::<Frac>::origin()),
                tau.coords.scale(frac!(2)),
            ),
            SymmOp::Translation(tau) => (SymmOp::Identity, *tau),
            SymmOp::Rotation(_r) => (*self, Vector3::zero()),
            SymmOp::Rotoinversion(_r) => (*self, Vector3::zero()),
            SymmOp::Screw(r, direct, screw) => {
                let tau = r.axis.dir().scaled_vec(r.kind.as_frac().abs() * (*screw));
                match direct {
                    RotationDirectness::Proper => (SymmOp::Rotation(*r), tau),
                    RotationDirectness::Improper => (SymmOp::Rotoinversion(*r), tau),
                }
            }
            SymmOp::Reflection(_) => (*self, Vector3::zero()),
            SymmOp::Glide(pl, tau) => (SymmOp::Reflection(*pl), *tau),
        }
    }

    /// The inverse symmetry operation.
    pub fn inv(&self) -> SymmOp {
        SymmOp::classify_affine(self.to_iso(false).inv()).unwrap()
    }

    /// Computes `self * rhs`: the operation of doing `rhs` and then doing `self`. Goes through a
    /// matrix, so is slow.
    pub fn compose_through_iso(&self, rhs: &Self) -> Self {
        SymmOp::classify_affine(self.to_iso(false) * rhs.to_iso(false)).unwrap()
    }

    /// Computes `self * rhs` without the conversion to an isometry.
    pub fn compose(&self, rhs: &Self) -> Self {
        match (*self, *rhs) {
            (SymmOp::Identity, rhs) => rhs,
            (lhs, SymmOp::Identity) => lhs,
            (SymmOp::Inversion(o1), SymmOp::Inversion(o2)) => {
                SymmOp::Translation((o1 - o2).scale(frac!(2)))
            }
            (SymmOp::Inversion(o), SymmOp::Translation(t)) => {
                SymmOp::Inversion(o - t.scale(frac!(1 / 2)))
            }
            (SymmOp::Inversion(o), SymmOp::Rotation(r)) => {
                if r.axis.contains(o) {
                    SymmOp::Rotoinversion(SimpleRotation {
                        axis: RotationAxis::new_with_dir(r.axis.dir(), o),
                        kind: r.kind,
                    })
                } else {
                    self.compose_through_iso(rhs)
                }
            }
            (SymmOp::Inversion(_), SymmOp::Rotoinversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Inversion(_), SymmOp::Screw(_, _, _)) => self.compose_through_iso(rhs),
            (SymmOp::Inversion(_), SymmOp::Reflection(_)) => self.compose_through_iso(rhs),
            (SymmOp::Inversion(_), SymmOp::Glide(_, _)) => self.compose_through_iso(rhs),
            (SymmOp::Translation(t), SymmOp::Inversion(o)) => {
                SymmOp::Inversion(o + t.scale(frac!(1 / 2)))
            }
            (SymmOp::Translation(t1), SymmOp::Translation(t2)) => SymmOp::Translation(t1 + t2),
            (SymmOp::Translation(_), SymmOp::Rotation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Translation(_), SymmOp::Rotoinversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Translation(_), SymmOp::Screw(_, _, _)) => self.compose_through_iso(rhs),
            (SymmOp::Translation(_), SymmOp::Reflection(_)) => self.compose_through_iso(rhs),
            (SymmOp::Translation(_), SymmOp::Glide(_, _)) => self.compose_through_iso(rhs),
            (SymmOp::Rotation(_), SymmOp::Inversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotation(_), SymmOp::Translation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotation(_), SymmOp::Rotation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotation(_), SymmOp::Rotoinversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotation(_), SymmOp::Screw(_, _, _)) => self.compose_through_iso(rhs),
            (SymmOp::Rotation(_), SymmOp::Reflection(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotation(_), SymmOp::Glide(_, _)) => self.compose_through_iso(rhs),
            (SymmOp::Rotoinversion(_), SymmOp::Inversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotoinversion(_), SymmOp::Translation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotoinversion(_), SymmOp::Rotation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotoinversion(_), SymmOp::Rotoinversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotoinversion(_), SymmOp::Screw(_, _, _)) => self.compose_through_iso(rhs),
            (SymmOp::Rotoinversion(_), SymmOp::Reflection(_)) => self.compose_through_iso(rhs),
            (SymmOp::Rotoinversion(_), SymmOp::Glide(_, _)) => self.compose_through_iso(rhs),
            (SymmOp::Screw(_, _, _), SymmOp::Inversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Screw(_, _, _), SymmOp::Translation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Screw(_, _, _), SymmOp::Rotation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Screw(_, _, _), SymmOp::Rotoinversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Screw(_, _, _), SymmOp::Screw(_, _, _)) => self.compose_through_iso(rhs),
            (SymmOp::Screw(_, _, _), SymmOp::Reflection(_)) => self.compose_through_iso(rhs),
            (SymmOp::Screw(_, _, _), SymmOp::Glide(_, _)) => self.compose_through_iso(rhs),
            (SymmOp::Reflection(_), SymmOp::Inversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Reflection(_), SymmOp::Translation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Reflection(_), SymmOp::Rotation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Reflection(_), SymmOp::Rotoinversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Reflection(_), SymmOp::Screw(_, _, _)) => self.compose_through_iso(rhs),
            (SymmOp::Reflection(pl1), SymmOp::Reflection(pl2)) => {
                if pl1 == pl2 {
                    SymmOp::Identity
                } else {
                    self.compose_through_iso(rhs)
                }
            }
            (SymmOp::Reflection(_), SymmOp::Glide(_, _)) => self.compose_through_iso(rhs),
            (SymmOp::Glide(_, _), SymmOp::Inversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Glide(_, _), SymmOp::Translation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Glide(_, _), SymmOp::Rotation(_)) => self.compose_through_iso(rhs),
            (SymmOp::Glide(_, _), SymmOp::Rotoinversion(_)) => self.compose_through_iso(rhs),
            (SymmOp::Glide(_, _), SymmOp::Screw(_, _, _)) => self.compose_through_iso(rhs),
            (SymmOp::Glide(_, _), SymmOp::Reflection(_)) => self.compose_through_iso(rhs),
            (SymmOp::Glide(pl1, tau1), SymmOp::Glide(pl2, tau2)) => {
                if pl1 == pl2 {
                    SymmOp::Translation(tau1 + tau2)
                } else {
                    self.compose_through_iso(rhs)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Error, PartialEq)]
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

/// 3x3 determinant, implemented for `Frac`.
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
            Isometry::tabled_matrix(m),
            det_scaled as f64 / ((den * den * den) as f64)
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
                Ok(Self::Identity)
            } else {
                Ok(Self::Translation(w))
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

            // scale axis by determinant, because geometric operations for rotoinversions describe
            // the rotation and not the inversion!
            let axis = (0..3)
                .filter_map(|c| {
                    if Y.column(c).abs().sum().is_zero() {
                        None
                    } else {
                        Some(Y.column(c))
                    }
                })
                .next()
                .unwrap()
                .clone_owned()
                * det;

            let rot_kind = if order > 2 {
                // (c) sense of rotation

                // the sense is, where u is the axis, x is some vector not parallel to u, and d =
                // det W

                // | u1 x1 (dWx)1 |
                // | u2 x2 (dWx)2 |
                // | u3 x3 (dWx)3 |

                // We can pick x to be one of the basis vectors.
                let sense: Frac = [
                    Vector3::x(),
                    Vector3::y(),
                    Vector3::x() - Vector3::z(),
                    Vector3::x() - Vector3::z(),
                    Vector3::x() + Vector3::y() + Vector3::z(),
                ]
                .into_iter()
                .filter_map(|x| {
                    let Z = Matrix3::from_columns(&[axis, x, W_abs * x]);
                    let z_det = det3x3(Z);
                    // println!("Z {Z} {z_det}");
                    if z_det.is_zero() {
                        None
                    } else {
                        Some(z_det)
                    }
                })
                .next()
                .unwrap_or_else(|| panic!("{} {}", axis, W));

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
                Some(RotationKind::new(is_ccw, rot_type.unsigned_abs() as usize))
            } else {
                // reflection
                Some(RotationKind::Two)
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
            for t_el in t.iter() {
                if t_el.numerator % order != 0 {
                    println!("W: {}", W);
                    println!("Y(±W): {}", Y);
                    println!("Y(W): {}", YW);
                    println!("w: {}", w);
                    println!("t: {}", t);
                    return Err(SymmOpError::NotSymmetryOperation(m));
                }
            }
            let w_g = t / frac!(order);
            // the location part of the translation, which isn't a screw or a glide
            let w_l = w - w_g;

            // println!("W: {}", W);
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

            // [W - I | I | -wl]

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
                (w_l.scale(frac!(-1))).column(0),
            ]);
            // println!("aug: {}", aug);
            let mut aug = aug.clone_owned();
            gauss_eliminate(&mut aug);
            // println!("aug eliminated: {aug}");

            // aug is now in its reduced form. If the first three entries in a row are 0, then we
            // check that the last entry is 0 (otherwise our system is unsolvable!), then we read
            // off a basis vector of the kernel from the second set of three rows. Otherwise, we can
            // solve for that row of the result by dividing out by the nonzero element.
            let mut kernel_basis = vec![];
            let mut center = Point3::origin();
            let mut do_cross = false;
            let mut sol = Vector3::zero();

            for row in 0..3 {
                if aug.fixed_view::<1, 3>(row, 0) != Vector3::zero().transpose() {
                    // part of solution
                    let mut sol_dim = 0;
                    while aug[(row, sol_dim)].is_zero() {
                        sol_dim += 1;
                    }
                    center[sol_dim] = aug[(row, 6)] / aug[(row, sol_dim)];
                    if !aug[(row, sol_dim + 1)].is_zero() && sol_dim < 2 {
                        // dbg!(row, sol_dim + 1);
                        // center[sol_dim + 1] = aug[(row, 6)] / aug[(row, sol_dim)];
                        sol = aug.fixed_view::<1, 3>(row, 0).clone_owned().transpose();
                    }
                } else if aug
                    .fixed_view::<1, 3>(row, 3)
                    .transpose()
                    .dot(&sol)
                    .is_zero()
                {
                    // part of kernel
                    // if this fails, the system has no solutions and we goofed
                    assert!(aug[(row, 6)].is_zero());
                    // add basis vector to kernel basis
                    kernel_basis.push(aug.fixed_view::<1, 3>(row, 3).transpose());
                } else if !sol.is_zero() {
                    do_cross = true;
                }
            }

            if do_cross {
                // dbg!(&sol, &kernel_basis, kernel_basis[0].cross(&sol));
                let mut other = Vector3::zeros();
                for (i, s) in sol.iter().enumerate() {
                    if s.is_zero() {
                        other[i] = frac!(1);
                        break;
                    }
                }
                if other.is_zero() {
                    panic!();
                }
                kernel_basis.push(other.cross(&sol));
            }

            // We already covered the cases where the kernel has dimension 3: identity and
            // translation.

            // dbg!(axis);

            // dbg!(&center);
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
    use nalgebra::vector;
    use pretty_assertions::assert_eq;
    use std::str::FromStr;

    use approx::assert_ulps_eq;

    use crate::{hermann_mauguin::PartialSymmOp, markup::ASCII};

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
    fn test_dir_vec3() {
        assert_eq!(
            Direction::new(vector![frac!(-1), frac!(1), frac!(-1)]).as_vec3(),
            Vector3::new(frac!(-1), frac!(1), frac!(-1))
        );

        assert_eq!(
            Direction::new(vector![frac!(1), frac!(-1), frac!(0)]).as_vec3(),
            Vector3::new(frac!(1), frac!(-1), frac!(0))
        );
    }

    #[test]
    fn test_weird_rots() {
        assert_eq!(
            ASCII
                .render_to_string(
                    &SymmOp::classify_affine(Isometry::from_str("z, x, y").unwrap()).unwrap()
                )
                .as_str(),
            "3+ \tx, x, x"
        );

        let op = SymmOp::classify_affine(Isometry::from_str("-z, -x+1/2, y+1/2").unwrap()).unwrap();

        dbg!(op);

        assert_eq!(
            ASCII.render_to_string(&op).as_str(),
            "3_2+ \tx -1/3, -x +1/6, -x"
        );
    }

    #[test]
    fn test_diagonal() {
        // let op = SymmOp::Rotation(SimpleRotation::new(
        //     RotationAxis::new(
        //         vector![frac!(0), frac!(0), frac!(1)],
        //         Point3::<Frac>::new(frac!(2 / 3), frac!(-1 / 3), frac!(1)),
        //     ),
        //     RotationKind::PosThree,
        // ));

        let op1 = SymmOp::classify_affine(Isometry::from_str("x-1,y,z").unwrap()).unwrap();
        let op2 = SymmOp::classify_affine(Isometry::from_str("-y, x-y, z").unwrap()).unwrap();
        let op3 = op1.compose(&op2);

        println!("{op1:?} {op2:?}\n{op1} {op2}");
        println!("{op3:?}");

        let op4 = op3.modulo_unit_cell();
        println!("{op4:?}");
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
    fn test_hex_rot() {
        assert_eq!(
            SymmOp::classify_affine(Isometry::from_str("-y,-x,-z").unwrap()).unwrap(),
            SymmOp::Rotation(SimpleRotation {
                axis: RotationAxis::new(vector![frac!(-1), frac!(1), frac!(0)], Point3::origin()),
                kind: RotationKind::Two,
            })
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
                Point3::new(frac!(1 / 2), frac!(0), frac!(0)),
            ),
            Vector3::new(frac!(1 / 4), frac!(-1 / 4), frac!(1 / 4)),
        )
        .conventional();

        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(false),
            ans.to_iso(false),
            "{}\n{}",
            symm.to_iso(false).mat(),
            ans.to_iso(false).mat()
        )
    }

    #[test]
    fn test_rotation() {
        // example 1 from ITA 1.2.2.4
        let iso = Isometry::from_str("y+1/4, -x+1/4, z+3/4").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        let ans = SymmOp::new_generalized_rotation(
            RotationAxis::new(Vector3::z(), Point3::new(frac!(1 / 4), frac!(0), frac!(0))),
            RotationKind::NegFour,
            true,
            Vector3::new(frac!(0), frac!(0), frac!(3 / 4)),
        );

        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(false),
            ans.to_iso(false),
            "{}\n{}",
            symm.to_iso(false).mat(),
            ans.to_iso(false).mat()
        )
    }

    #[test]
    fn test_rotoinv_to_iso() {
        // example 2 from ITA 1.2.2.4
        let iso = Isometry::from_str("y, -z+1/2, x+1/2").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        let ans = SymmOp::new_generalized_rotation(
            RotationAxis::new(
                Vector3::new(frac!(1), frac!(-1), frac!(-1)),
                Point3::new(frac!(0), frac!(0), frac!(1 / 2)),
            ),
            RotationKind::NegThree,
            false,
            Vector3::zero(),
        );
        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(false),
            ans.to_iso(false),
            "{}\n{}",
            symm.to_iso(false).mat(),
            ans.to_iso(false).mat()
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
                Point3::new(frac!(0), frac!(1 / 2), frac!(1 / 2)),
            ),
            RotationKind::PosThree,
            false,
            Vector3::zero(),
        );
        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(false),
            ans.to_iso(false),
            "{}\n{}",
            symm.to_iso(false).mat(),
            ans.to_iso(false).mat()
        )
    }

    #[test]
    fn test_dir_screw_110() {
        let dir = Direction::new(vector![frac!(1), frac!(1), frac!(0)]);
        let screw = Vector3::new(frac!(-1 / 2), frac!(-1 / 2), frac!(0));
        assert_eq!(dir.compute_scale(screw), Some(frac!(-1 / 2)));
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

    #[test]
    fn test_hexagonal() {
        let iso = Isometry::from_str("-y, -x, -z+5/6").unwrap();
        let symm = SymmOp::classify_affine(iso).unwrap();
        let ans = SymmOp::new_generalized_rotation(
            RotationAxis::new(
                Vector3::new(frac!(1), frac!(-1), frac!(0)),
                Point3::new(frac!(0), frac!(0), frac!(5 / 12)),
            ),
            RotationKind::Two,
            true,
            Vector3::zero(),
        );
        assert_eq!(symm, ans, "{:?} {:?}", symm, ans);
        assert_eq!(
            symm.to_iso(false),
            ans.to_iso(false),
            "{}\n{}",
            symm.to_iso(false).mat(),
            ans.to_iso(false).mat()
        )
    }

    #[test]
    fn classify_round_trip() {
        for op in many_symmops() {
            // assert_eq!(SymmOp::classify_affine(op.to_iso(false)), Ok(op));
            // println!("{op:?}");
            for is_hex in [true, false] {
                let m2 = op.to_iso(is_hex);
                // println!("{m2}");
                let m1 = SymmOp::classify_affine(op.to_iso(is_hex))
                    .unwrap()
                    .to_iso(is_hex);

                assert_eq!(m2, m1, "{m2} {m1} {op:?} {is_hex}");
            }
        }
    }

    #[test]
    fn test_inv_compose() {
        let inv = SymmOp::Inversion(Point3::origin());
        let tau = SymmOp::Translation(Vector3::new(frac!(1 / 2), frac!(1 / 2), frac!(1 / 2)));
        let inv2 = SymmOp::Inversion(Point3::new(frac!(1 / 4), frac!(1 / 4), frac!(1 / 4)));

        assert_eq!(tau.compose(&inv), inv2);
    }

    #[test]
    fn test_rot_tau() {
        for op in many_symmops() {
            let (rot, tau) = op.to_rot_and_tau();
            // dbg!(op);
            let tau = SymmOp::Translation(tau);
            // println!("rot {}\ntau {}", rot.to_iso(false).mat(), tau.to_iso(false).mat());
            // println!("combo {}", (tau.to_iso(false) * rot.to_iso(false)).mat());
            let tau_rot = tau.compose(&rot);
            // assert_eq!(tau_rot, op)
            let (m1, m2) = (tau_rot.to_iso(false).mat(), op.to_iso(false).mat());
            assert_eq!(m1, m2, "{m1} {m2}");
        }
    }

    #[test]
    fn hard_iso_classification() {
        SymmOp::classify_affine(Isometry::from_str("-y-1/3,-z-1/3,x+1/3").unwrap()).unwrap();
    }

    #[test]
    fn test_iso_display() {
        for op in many_symmops() {
            // println!("{:?}", op);
            // println!("{}", op.to_iso(false));
            // println!("{}", op.to_iso(false).inv());
            assert_eq!(
                Isometry::from_str(ASCII.render_to_string(&op.to_iso(false)).as_str()).unwrap(),
                op.to_iso(false),
                "{}\n{}\n{}",
                ASCII.render_to_string(&op.to_iso(false)),
                Isometry::from_str(ASCII.render_to_string(&op.to_iso(false)).as_str())
                    .unwrap()
                    .mat(),
                op.to_iso(false).mat()
            )
        }
    }

    // #[test]
    // fn test_dir_decomposition() {
    //     for op in many_symmops() {
    //         if let Some(dir) = op.symmetry_direction() {
    //             if op.translation_component().is_none() {
    //                 if let Some(partial) = PartialSymmOp::try_from_op(&op) {
    //                     let mut op1 = partial.to_symmop_with_dir(dir);
    //                     if op1.to_iso(false).modulo_unit_cell() != op.to_iso(false).modulo_unit_cell() {
    //                         op1 = op1.inv();
    //                     }
    //                     assert_eq!(
    //                         op1.to_iso(false).modulo_unit_cell(),
    //                         op.to_iso(false).modulo_unit_cell(),
    //                         "\n{:#?}\n{:#?}\n{:?} {:?}",
    //                         op1,
    //                         op,
    //                         partial,
    //                         dir
    //                     );
    //                 }
    //             }
    //         }
    //     }
    // }

    fn many_symmops() -> Vec<SymmOp> {
        let mut ops = vec![];
        ops.extend_from_slice(&hard_symmops());
        ops.extend_from_slice(&many_symmops_base());
        let ops_inv: Vec<SymmOp> = ops
            .iter()
            .map(|o| {
                // dbg!(o, o.conventional());
                // println!("{}", o.to_iso(false));
                // println!("{}", o.to_iso(false).inv());
                o.inv()
            })
            .collect();
        ops.extend(ops_inv);
        ops
    }

    fn hard_symmops() -> Vec<SymmOp> {
        vec!["-x+y,y,-z+1/2"]
            .into_iter()
            .map(|s| SymmOp::classify_affine(Isometry::from_str(s).unwrap()).unwrap())
            .collect()
    }

    fn many_symmops_base() -> Vec<SymmOp> {
        vec![
            SymmOp::Identity,
            SymmOp::Inversion(Point3::<Frac>::origin()),
            SymmOp::Inversion(Point3::<Frac>::new(
                frac!(1 / 2),
                frac!(1 / 4),
                frac!(-1 / 2),
            )),
            SymmOp::Rotation(SimpleRotation::new(
                RotationAxis::new(
                    vector![frac!(0), frac!(0), frac!(1)],
                    Point3::<Frac>::new(frac!(0), frac!(0), frac!(1 / 3)),
                ),
                RotationKind::PosThree,
            )),
            SymmOp::Rotation(SimpleRotation::new(
                RotationAxis::new(
                    vector![frac!(1), frac!(-1), frac!(0)],
                    Point3::<Frac>::new(frac!(1 / 2), frac!(-1 / 2), frac!(0)),
                ),
                RotationKind::Two,
            )),
            SymmOp::Rotation(SimpleRotation::new(
                RotationAxis::new(
                    vector![frac!(0), frac!(1), frac!(0)],
                    Point3::<Frac>::new(frac!(0), frac!(2), frac!(0)),
                ),
                RotationKind::NegFour,
            )),
            SymmOp::Rotation(SimpleRotation::new(
                RotationAxis::new(
                    vector![frac!(0), frac!(0), frac!(1)],
                    Point3::<Frac>::new(frac!(0), frac!(0), frac!(1)),
                ),
                RotationKind::NegSix,
            )),
            SymmOp::Rotation(SimpleRotation::new(
                RotationAxis::new(
                    vector![frac!(0), frac!(0), frac!(1)],
                    Point3::<Frac>::new(frac!(0), frac!(0), frac!(1)),
                ),
                RotationKind::PosThree,
            )),
            SymmOp::Rotoinversion(SimpleRotation::new(
                RotationAxis::new(
                    vector![frac!(0), frac!(1), frac!(0)],
                    Point3::<Frac>::new(frac!(0), frac!(2), frac!(0)),
                ),
                RotationKind::NegFour,
            )),
            SymmOp::Rotoinversion(SimpleRotation::new(
                RotationAxis::new(
                    vector![frac!(0), frac!(0), frac!(1)],
                    Point3::<Frac>::new(frac!(0), frac!(0), frac!(1)),
                ),
                RotationKind::NegSix,
            )),
            SymmOp::Screw(
                SimpleRotation::new(
                    RotationAxis::new(
                        vector![frac!(0), frac!(0), frac!(1)],
                        Point3::<Frac>::new(frac!(0), frac!(0), frac!(1 / 3)),
                    ),
                    RotationKind::PosThree,
                ),
                RotationDirectness::Proper,
                frac!(1 / 3),
            ),
            SymmOp::Screw(
                SimpleRotation::new(
                    RotationAxis::new(
                        vector![frac!(0), frac!(1), frac!(0)],
                        Point3::<Frac>::new(frac!(0), frac!(2), frac!(0)),
                    ),
                    RotationKind::NegFour,
                ),
                RotationDirectness::Improper,
                frac!(-3 / 4),
            ),
            SymmOp::Screw(
                SimpleRotation::new(
                    RotationAxis::new(
                        vector![frac!(0), frac!(0), frac!(1)],
                        Point3::<Frac>::new(frac!(0), frac!(0), frac!(0)),
                    ),
                    RotationKind::NegSix,
                ),
                RotationDirectness::Improper,
                frac!(5 / 6),
            ),
            SymmOp::Reflection(Plane::from_basis_and_origin(
                vector![frac!(0), frac!(0), frac!(1)],
                vector![frac!(2), frac!(1), frac!(0)],
                Point3::origin(),
            )),
            SymmOp::Reflection(Plane::from_basis_and_origin(
                vector![frac!(0), frac!(0), frac!(1)],
                vector![frac!(1), frac!(0), frac!(0)],
                Point3::origin(),
            )),
            SymmOp::Reflection(Plane::from_basis_and_origin(
                vector![frac!(0), frac!(0), frac!(1)],
                vector![frac!(1), frac!(2), frac!(0)],
                Point3::new(frac!(0), frac!(0), frac!(0)),
            )),
            SymmOp::Glide(
                Plane::from_basis_and_origin(
                    vector![frac!(0), frac!(0), frac!(1)],
                    vector![frac!(1), frac!(-1), frac!(0)],
                    Point3::new(frac!(0), frac!(0), frac!(0)),
                ),
                vector![frac!(0), frac!(0), frac!(1 / 2)],
            ),
            SymmOp::Glide(
                Plane::from_basis_and_origin(
                    vector![frac!(0), frac!(0), frac!(1)],
                    vector![frac!(1), frac!(1), frac!(0)],
                    Point3::new(frac!(1), frac!(0), frac!(0)),
                ),
                vector![frac!(1 / 4), frac!(1 / 4), frac!(0)],
            ),
            SymmOp::Glide(
                Plane::from_basis_and_origin(
                    vector![frac!(0), frac!(0), frac!(1)],
                    vector![frac!(1), frac!(1), frac!(0)],
                    Point3::new(frac!(0), frac!(0), frac!(3 / 8)),
                ),
                vector![frac!(3 / 4), frac!(3 / 4), frac!(0)],
            ),
        ]
    }
}
