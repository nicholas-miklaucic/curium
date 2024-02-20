//! Utilities for using the Hermann-Mauguin notation.

use nalgebra::{Point3, Vector3};
use num_traits::Zero;

use crate::symbols::*;
use crate::symmop::{Direction, Plane, RotationAxis, RotationKind, SymmOp};
use crate::{
    frac,
    markup::{Block, RenderBlocks},
    symmop::ScrewOrder,
};
use std::cmp::PartialOrd;

/// The order of a rotation, including 1.
pub type RotOrder = i8;

/// A description of a symmetry operation with an already-known symmetry direction: essentially, one
/// part of a Hermann-Mauguin symbol. See Table 2.1.2.1.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum PartialSymmOp {
    /// A glide with vector a/2, denoted `a`.
    AGlide,
    /// A glide with vector b/2, denoted `b`.
    BGlide,
    /// A glide with vector c/2, denoted `c`.
    CGlide,
    /// Two different operations: glides with vectors 1/2 of the lattice vectors not parallel to the
    /// normal vector of the plane. Denoted `e`.
    EGlide,
    /// A 'diagonal' glide, with a glide vector essentially in as many lattice directions as
    /// possible given the direction of the plane. Denoted `n`.
    NGlide,
    /// A 'diamond' glide, representing two operations with alternating signs roughly akin to 1/4 of
    /// the n glide. Denoted `d`.
    DGlide,
    /// A rotation/screw. The order must be plus or minus 1, 2, 3, 4, or 6. The screw order
    /// should be between 0 and |k| - 1, where k is the order.
    GenRotation(RotOrder, ScrewOrder),
}

impl RenderBlocks for PartialSymmOp {
    fn components(&self) -> Vec<Block> {
        match *self {
            PartialSymmOp::AGlide => vec![A_GLIDE],
            PartialSymmOp::BGlide => vec![B_GLIDE],
            PartialSymmOp::CGlide => vec![C_GLIDE],
            PartialSymmOp::EGlide => vec![E_GLIDE],
            PartialSymmOp::NGlide => vec![N_GLIDE],
            PartialSymmOp::DGlide => vec![D_GLIDE],
            PartialSymmOp::GenRotation(-2, f) if f.is_zero() => vec![MIRROR],
            PartialSymmOp::GenRotation(r, f) if f.is_zero() => {
                vec![Block::new_int(r as i64)]
            }
            PartialSymmOp::GenRotation(r, s) => {
                vec![Block::Subscript(
                    Block::new_int(r as i64).into(),
                    Block::Blocks((s * frac!(r)).components()).into(),
                )]
            }
        }
    }
}

impl PartialSymmOp {
    /// Converts from a `SymmOp`, returning a `PartialSymmOp` if it exists and `None` otherwise.
    pub fn try_from_op(op: &SymmOp) -> Option<PartialSymmOp> {
        match *op {
            SymmOp::Identity => Some(PartialSymmOp::GenRotation(1, frac!(0))),
            SymmOp::Inversion(tau) => {
                if tau == Point3::origin() {
                    Some(Self::GenRotation(-1, frac!(0)))
                } else {
                    None
                }
            }
            SymmOp::Translation(_) => None,
            SymmOp::Rotation(rot) => {
                Some(Self::GenRotation(rot.kind.order() as RotOrder, frac!(0)))
            }
            SymmOp::Rotoinversion(rot) => {
                Some(Self::GenRotation(-rot.kind.order() as RotOrder, frac!(0)))
            }
            SymmOp::Screw(rot, dir, tau) => Some(Self::GenRotation(
                (rot.kind.order() as RotOrder) * dir.det_sign() as RotOrder,
                tau,
            )),
            SymmOp::Reflection(_pl) => Some(Self::GenRotation(-2, frac!(0))),
            SymmOp::Glide(pl, tau) => {
                // table 2.1.2.1
                let [a, b, c] = *tau.as_slice() else {
                    panic!("Tau is 3D")
                };
                let f0 = frac!(0);
                let f14 = frac!(1 / 4);
                let f12 = frac!(1 / 2);

                if [a, b, c] == [f12, f0, f0] {
                    Some(Self::AGlide)
                } else if [a, b, c] == [f0, f12, f0] {
                    Some(Self::BGlide)
                } else if [a, b, c] == [f0, f0, f12] {
                    Some(Self::CGlide)
                } else {
                    // the remaining cases have many versions depending on the orientation, so
                    // it's most useful to work with generic basis vectors in the plane
                    let (v1, v2) = pl.basis_vectors();
                    let v_abc = Vector3::new(a, b, c);
                    if (v1 + v2).scale(f12) == v_abc {
                        Some(Self::NGlide)
                    } else if (v1 - v2).scale(f14) == v_abc || (v1 + v2).scale(f14) == v_abc {
                        Some(Self::DGlide)
                    } else if v1.scale(f12) == v_abc || v2.scale(f12) == v_abc {
                        Some(Self::EGlide)
                    } else {
                        None
                    }
                }
            }
        }
    }

    pub fn is_reflection(&self) -> bool {
        match *self {
            PartialSymmOp::GenRotation(-2, f) if f.is_zero() => true,
            PartialSymmOp::GenRotation(_, _) => false,
            _ => true,
        }
    }

    /// Priority under the rules in page 44 of ITA: essentially, higher-priority symbols appear in
    /// the HM symbol if there's a choice. The rules don't specify this, but clearly higher-order
    /// rotations take priority over lower-order ones and lower-order screws take priority over
    /// higher-order ones. Rotoinversions are lower-priority than rotations.
    pub fn priority(&self) -> i8 {
        match *self {
            PartialSymmOp::AGlide => 10,
            PartialSymmOp::BGlide => 10,
            PartialSymmOp::CGlide => 10,
            PartialSymmOp::EGlide => 15,
            PartialSymmOp::NGlide => 5,
            PartialSymmOp::DGlide => 10, // not specified?
            PartialSymmOp::GenRotation(-2, f) if f.is_zero() => 20,
            PartialSymmOp::GenRotation(1, f) if f.is_zero() => 1,
            PartialSymmOp::GenRotation(-1, f) if f.is_zero() => -1,
            PartialSymmOp::GenRotation(r, s) => {
                if r > 0 {
                    r * 10 - s.numerator as i8
                } else {
                    -r * 10 - s.numerator as i8 - 10
                }
            }
        }
    }

    /// Combines the two operations. Returns None if the two operations can't be compared. Note
    /// that, due to the double e-glide, this does not strictly return one of the inputs, because
    /// max(a, b) = e in some settings.
    pub fn partial_max(&self, rhs: &Self, use_e_glide: bool) -> Option<Self> {
        match (*self, *rhs, use_e_glide) {
            (
                PartialSymmOp::AGlide | PartialSymmOp::BGlide | PartialSymmOp::CGlide,
                PartialSymmOp::AGlide | PartialSymmOp::BGlide | PartialSymmOp::CGlide,
                true,
            ) => {
                if self == rhs {
                    Some(*self) // just the same
                } else {
                    // this is the double-glide case
                    Some(PartialSymmOp::EGlide)
                }
            }
            (lhs, rhs, _) => lhs
                .partial_cmp(&rhs)
                .map(|o| if o.is_ge() { *self } else { rhs }),
        }
    }

    /// Converts to a full `SymmOp` using the given symmetry direction.
    pub fn to_symmop_with_dir(&self, dir: Direction) -> SymmOp {
        match *self {
            Self::GenRotation(1, f) if f.is_zero() => SymmOp::Identity,
            Self::GenRotation(-1, f) if f.is_zero() => SymmOp::Inversion(Point3::origin()),
            Self::GenRotation(r, s) if (r, s) != (-2, frac!(0)) => {
                let axis = RotationAxis::new(dir.as_vec3(), Point3::origin());
                let tau = axis.dir.scaled_vec(s);
                // dbg!(tau);
                let kind = RotationKind::new(true, r.unsigned_abs() as usize);
                SymmOp::new_generalized_rotation(axis, kind, r.is_positive(), tau)
            }
            _ => {
                let f0 = frac!(0);
                let f14 = frac!(1 / 4);
                let f12 = frac!(1 / 2);

                // if [a, b, c] == [f12, f0, f0] {
                //     Some(Self::AGlide)
                // } else if [a, b, c] == [f0, f12, f0] {
                //     Some(Self::BGlide)
                // } else if [a, b, c] == [f0, f0, f12] {
                //     Some(Self::CGlide)
                // } else {
                //     // the remaining cases have many versions depending on the orientation, so
                //     // it's most useful to work with generic basis vectors in the plane
                //     let (b1, b2) = pl.basis();
                //     let v1 = b1.as_vec3();
                //     let v2 = b2.as_vec3();
                //     let v_abc = Vector3::new(a, b, c);
                //     if (v1 + v2).scale(f12) == v_abc {
                //         Some(Self::NGlide)
                //     } else if (v1 - v2).scale(f14) == v_abc {
                //         Some(Self::DGlide)
                //     } else if (v1 + v2).scale(f14) == v_abc {
                //         Some(Self::DGlide)
                //     } else if v1.scale(f12) == v_abc {
                //         Some(Self::EGlide)
                //     } else if v2.scale(f12) == v_abc {
                //         Some(Self::EGlide)
                //     } else {
                //         None
                //     }
                // }
                let (d1, d2) = dir.plane_basis();
                let (v1, v2) = (d1.as_vec3(), d2.as_vec3());
                let tau = match *self {
                    PartialSymmOp::AGlide => Vector3::new(f12, f0, f0),
                    PartialSymmOp::BGlide => Vector3::new(f0, f12, f0),
                    PartialSymmOp::CGlide => Vector3::new(f0, f0, f12),
                    // TODO do we need all of the potential operations here? Can a symbol be
                    // mistakenly excluded because one of its generators was redundant?
                    PartialSymmOp::EGlide => v1.scale(f12),
                    PartialSymmOp::NGlide => (v1 + v2).scale(f12),
                    PartialSymmOp::DGlide => (v1 + v2).scale(f14),
                    PartialSymmOp::GenRotation(-2, f) if f.is_zero() => Vector3::zero(),
                    _ => unreachable!(),
                };

                let plane = Plane::from_basis_and_origin(v1, v2, Point3::origin());
                SymmOp::new_generalized_reflection(plane, tau)
            }
        }
    }

    /// The order of the rotation, ignoring any screws or inversions.
    pub fn rot_kind(&self) -> RotOrder {
        match *self {
            PartialSymmOp::AGlide
            | PartialSymmOp::BGlide
            | PartialSymmOp::CGlide
            | PartialSymmOp::EGlide
            | PartialSymmOp::NGlide
            | PartialSymmOp::DGlide => 2,
            PartialSymmOp::GenRotation(r, _s) => r.abs(),
        }
    }
}

impl PartialOrd for PartialSymmOp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_reflection() == other.is_reflection() {
            self.priority().partial_cmp(&other.priority())
        } else {
            None
        }
    }
}

/// A full Hermann-Mauguin symbol component, describing the rotation and reflection in a direction.
/// Examples are `2/m`, `2_1 / n`, `c`, and `2`.
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct FullHMSymbolUnit {
    /// Rotation component.
    pub rotation: Option<PartialSymmOp>,
    /// Reflection component.
    pub reflection: Option<PartialSymmOp>,
    // TODO remove this
    // Keeps track of ops, for debugging.
    // ops: Vec<PartialSymmOp>,
}

impl FullHMSymbolUnit {
    /// Updates the symbol to summarize the previous symbol and the new operation.
    /// - If the new operation is a rotation/reflection and the current symbol has no previous
    ///   operation of that type, the new operation enters that spot.
    /// - If the new operation is with an existing operation, then the priority rules described in
    ///   page 44 of ITA are applied. The 'larger' is chosen, where m > e > a, b, c > n and
    ///   rotations > screws. Two axial glide planes can combine to form e, the double glide plane.
    pub fn and(&mut self, op: PartialSymmOp, use_e_glide: bool) {
        if op.is_reflection() {
            self.reflection = self.reflection.as_ref().map_or_else(
                || Some(op),
                |r| r.partial_max(&op, use_e_glide).or(Some(op)),
            );
        } else {
            self.rotation = self.rotation.as_ref().map_or_else(
                || Some(op),
                |r| r.partial_max(&op, use_e_glide).or(Some(op)),
            );
        }
        // self.ops.push(op.clone());
    }

    /// Gets the rotation component, or the reflection, or the identity.
    pub fn first_op(&self) -> PartialSymmOp {
        self.rotation
            .or(self.reflection)
            .unwrap_or(PartialSymmOp::GenRotation(1, frac!(0)))
    }

    /// Gets the symmetry operations.
    pub fn ops(&self) -> Vec<PartialSymmOp> {
        [self.rotation.as_slice(), self.reflection.as_slice()].concat()
    }

    pub const A: FullHMSymbolUnit = FullHMSymbolUnit {
        rotation: None,
        reflection: Some(PartialSymmOp::AGlide),
    };

    pub const B: FullHMSymbolUnit = FullHMSymbolUnit {
        rotation: None,
        reflection: Some(PartialSymmOp::BGlide),
    };

    pub const C: FullHMSymbolUnit = FullHMSymbolUnit {
        rotation: None,
        reflection: Some(PartialSymmOp::CGlide),
    };

    pub const M: FullHMSymbolUnit = FullHMSymbolUnit {
        rotation: None,
        reflection: Some(PartialSymmOp::GenRotation(-2, frac!(0))),
    };
}

impl RenderBlocks for FullHMSymbolUnit {
    fn components(&self) -> Vec<Block> {
        match (&self.rotation, &self.reflection) {
            (Some(r1), Some(r2)) => {
                vec![Block::Fraction(
                    Block::Blocks(r1.components()).into(),
                    Block::Blocks(r2.components()).into(),
                )]
            }
            (Some(r), None) | (None, Some(r)) => r.components(),
            (None, None) => vec![Block::new_int(1)],
        }
    }
}
