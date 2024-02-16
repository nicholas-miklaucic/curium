//! Describes space group symmetry in enough detail to perform the calculations necessary to derive
//! the information tabulated in ITA and other libraries. Unlike many other libraries, which require
//! storing all of this data per group, this information is intended for use only when needed. The
//! generic [`SpaceGroup`], a simple 230-element enum, is all that is needed to describe a space
//! group for serialization.

use std::str::FromStr;

use nalgebra::{Point3, Vector3};
use num_traits::Zero;

use crate::{
    algebra::{generate_elements, FiniteGroup, FinitelyGeneratedGroup, Group},
    frac,
    fract::Frac,
    group_classes::CrystalSystem,
    hall::HallCenteringType,
    lattice::{CenteringType, LatticeSystem},
    markup::{Block, RenderBlocks, ITA},
    symbols::{A_GLIDE, B_GLIDE, C_GLIDE, D_GLIDE, E_GLIDE, MIRROR, N_GLIDE},
    symmop::{Direction, Plane, RotationAxis, RotationKind, ScrewOrder, SymmOp},
};

/// The short Hermann-Mauguin symbols in order, represented using ASCII, no underscores for screws,
/// and with hexagonal axes for the rhombohedral groups.
pub const SPACE_GROUP_SYMBOLS: [&'static str; 230] = [
    "P1", "P-1", "P2", "P21", "C2", "Pm", "Pc", "Cm", "Cc", "P2/m", "P21/m", "C2/m", "P2/c",
    "P21/c", "C2/c", "P222", "P2221", "P21212", "P212121", "C2221", "C222", "F222", "I222",
    "I212121", "Pmm2", "Pmc21", "Pcc2", "Pma2", "Pca21", "Pnc2", "Pmn21", "Pba2", "Pna21", "Pnn2",
    "Cmm2", "Cmc21", "Ccc2", "Amm2", "Aem2", "Ama2", "Aea2", "Fmm2", "Fdd2", "Imm2", "Iba2",
    "Ima2", "Pmmm", "Pnnn", "Pccm", "Pban", "Pmma", "Pnna", "Pmna", "Pcca", "Pbam", "Pccn", "Pbcm",
    "Pnnm", "Pmmn", "Pbcn", "Pbca", "Pnma", "Cmcm", "Cmce", "Cmmm", "Cccm", "Cmme", "Ccce", "Fmmm",
    "Fddd", "Immm", "Ibam", "Ibca", "Imma", "P4", "P41", "P42", "P43", "I4", "I41", "P-4", "I-4",
    "P4/m", "P42/m", "P4/n", "P42/n", "I4/m", "I41/a", "P422", "P4212", "P4122", "P41212", "P4222",
    "P42212", "P4322", "P43212", "I422", "I4122", "P4mm", "P4bm", "P42cm", "P42nm", "P4cc", "P4nc",
    "P42mc", "P42bc", "I4mm", "I4cm", "I41md", "I41cd", "P-42m", "P-42c", "P-421m", "P-421c",
    "P-4m2", "P-4c2", "P-4b2", "P-4n2", "I-4m2", "I-4c2", "I-42m", "I-42d", "P4/mmm", "P4/mcc",
    "P4/nbm", "P4/nnc", "P4/mbm", "P4/mnc", "P4/nmm", "P4/ncc", "P42/mmc", "P42/mcm", "P42/nbc",
    "P42/nnm", "P42/mbc", "P42/mnm", "P42/nmc", "P42/ncm", "I4/mmm", "I4/mcm", "I41/amd",
    "I41/acd", "P3", "P31", "P32", "R3", "P-3", "R-3", "P312", "P321", "P3112", "P3121", "P3212",
    "P3221", "R32", "P3m1", "P31m", "P3c1", "P31c", "R3m", "R3c", "P-31m", "P-31c", "P-3m1",
    "P-3c1", "R-3m", "R-3c", "P6", "P61", "P65", "P62", "P64", "P63", "P-6", "P6/m", "P63/m",
    "P622", "P6122", "P6522", "P6222", "P6422", "P6322", "P6mm", "P6cc", "P63cm", "P63mc", "P-6m2",
    "P-6c2", "P-62m", "P-62c", "P6/mmm", "P6/mcc", "P63/mcm", "P63/mmc", "P23", "F23", "I23",
    "P213", "I213", "Pm-3", "Pn-3", "Fm-3", "Fd-3", "Im-3", "Pa-3", "Ia-3", "P432", "P4232",
    "F432", "F4132", "I432", "P4332", "P4132", "I4132", "P-43m", "F-43m", "I-43m", "P-43n",
    "F-43c", "I-43d", "Pm-3m", "Pn-3n", "Pm-3n", "Pn-3m", "Fm-3m", "Fm-3c", "Fd-3m", "Fd-3c",
    "Im-3m", "Ia-3d",
];

/// A space group with a specific choice of symmetry directions. Essentially corresponds to a full
/// Hall symbol.
#[derive(Debug, Clone)]
pub struct SpaceGroupSetting {
    /// The coordinate system.
    pub lattice_type: LatticeSystem,
    /// The centering type of the lattice, defining the translational symmetry of the spacegroup.
    pub centering: HallCenteringType,
    /// The generator operations, not including pure translations.
    pub linear_generators: Vec<SymmOp>,
    /// The symmetry operations that define the group beyond translational symmetry. Should form a
    /// complete subgroup in the quotient group of the space group and the translational subgroup:
    /// i.e., when combining operations, the result should be a translated version of another
    /// operation.
    pub symmops: Vec<SymmOp>, // TODO this needs to properly account for centering type
}

/// Implements symmetry operation composition in the quotient group of the translational subgroup of
/// the group. More prosaically, this adds a mod 1 to everything: two points are equivalent if their
/// difference is integral.
impl Group<SymmOp> for SpaceGroupSetting {
    fn identity(&self) -> SymmOp {
        SymmOp::Identity
    }

    fn inv(&self, element: &SymmOp) -> SymmOp {
        self.residue(&element.inv())
    }

    fn compose(&self, a: &SymmOp, b: &SymmOp) -> SymmOp {
        // println!("{a} * {b} = ");
        // println!(
        //     " {} * {} = ",
        //     ITA.render_to_string(&a.to_iso()),
        //     ITA.render_to_string(&b.to_iso())
        // );
        // println!(
        //     "{}",
        //     ITA.render_to_string(&(a.to_iso() * b.to_iso()).modulo_unit_cell())
        // );

        // We have to reduce the isometry modulo a unit cell *before* we convert to a SymmOp. The
        // reason is that our Fracs can't handle all of the potential SymmOps in the total space of
        // operations, but we can represent all of the SymmOps we actually need. For example,
        // combining a rotation y, z, x with t(0, 0, 1) seems harmless, but the resulting axis is
        // quite weird.
        let el = SymmOp::classify_affine((a.to_iso() * b.to_iso()).modulo_unit_cell()).unwrap();
        // println!("el: {el}");
        self.residue(&el)
    }

    fn equiv(&self, a: &SymmOp, b: &SymmOp) -> bool {
        self.residue(&a) == self.residue(&b)
    }

    fn residue(&self, el: &SymmOp) -> SymmOp {
        el.modulo_unit_cell()
    }
}

impl FinitelyGeneratedGroup<SymmOp> for SpaceGroupSetting {
    type Generators = Vec<SymmOp>;

    fn generators(&self) -> Self::Generators {
        let mut gens = self.centering.centering_ops();
        gens.extend(self.linear_generators.clone());
        gens
    }
}

impl FiniteGroup<SymmOp> for SpaceGroupSetting {
    type Elements = Vec<SymmOp>;

    fn elements(&self) -> Self::Elements {
        self.symmops.clone()
    }
}

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
            PartialSymmOp::GenRotation(-2, 0) => vec![MIRROR],
            PartialSymmOp::GenRotation(r, 0) => {
                vec![Block::new_int(r as i64)]
            }
            PartialSymmOp::GenRotation(r, s) => {
                vec![Block::Subscript(
                    Block::new_int(r as i64).into(),
                    Block::new_uint(s.rem_euclid(r) as u64).into(),
                )]
            }
        }
    }
}

impl PartialSymmOp {
    /// Converts from a `SymmOp`, returning a `PartialSymmOp` if it exists and `None` otherwise.
    pub fn try_from_op(op: &SymmOp) -> Option<PartialSymmOp> {
        match *op {
            SymmOp::Identity => Some(PartialSymmOp::GenRotation(1, 0)),
            SymmOp::Inversion(tau) => {
                if tau == Point3::origin() {
                    Some(Self::GenRotation(-1, 0))
                } else {
                    None
                }
            }
            SymmOp::Translation(_) => None,
            SymmOp::Rotation(rot) => Some(Self::GenRotation(rot.kind.order() as RotOrder, 0)),
            SymmOp::Rotoinversion(rot) => Some(Self::GenRotation(-rot.kind.order() as RotOrder, 0)),
            SymmOp::Screw(rot, dir, tau) => Some(Self::GenRotation(
                (rot.kind.order() as RotOrder) * dir.det_sign() as RotOrder,
                tau.rem_euclid(rot.kind.order() as ScrewOrder),
            )),
            SymmOp::Reflection(_pl) => Some(Self::GenRotation(-2, 0)),
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
                    } else if (v1 - v2).scale(f14) == v_abc {
                        Some(Self::DGlide)
                    } else if (v1 + v2).scale(f14) == v_abc {
                        Some(Self::DGlide)
                    } else if v1.scale(f12) == v_abc {
                        Some(Self::EGlide)
                    } else if v2.scale(f12) == v_abc {
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
            PartialSymmOp::GenRotation(-2, 0) => true,
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
            PartialSymmOp::GenRotation(-2, 0) => 20,
            PartialSymmOp::GenRotation(1, 0) => 1,
            PartialSymmOp::GenRotation(-1, 0) => -1,
            PartialSymmOp::GenRotation(r, s) => {
                if r > 0 {
                    r * 10 - s
                } else {
                    -r * 10 - s - 10
                }
            }
        }
    }

    /// Combines the two operations. Returns None if the two operations can't be compared. Note
    /// that, due to the double e-glide, this does not strictly return one of the inputs, because
    /// max(a, b) = e.
    pub fn partial_max(&self, rhs: &Self) -> Option<Self> {
        match (self.clone(), rhs.clone()) {
            (
                PartialSymmOp::AGlide | PartialSymmOp::BGlide | PartialSymmOp::CGlide,
                PartialSymmOp::AGlide | PartialSymmOp::BGlide | PartialSymmOp::CGlide,
            ) => {
                if self == rhs {
                    Some(self.clone()) // just the same
                } else {
                    // this is the double-glide case
                    Some(PartialSymmOp::EGlide)
                }
            }
            (lhs, rhs) => {
                lhs.partial_cmp(&rhs)
                    .map(|o| if o.is_ge() { self.clone() } else { rhs.clone() })
            }
        }
    }

    /// Converts to a full `SymmOp` using the given symmetry direction.
    pub fn to_symmop_with_dir(&self, dir: Direction) -> SymmOp {
        match *self {
            Self::GenRotation(1, 0) => SymmOp::Identity,
            Self::GenRotation(-1, 0) => SymmOp::Inversion(Point3::origin()),
            Self::GenRotation(r, s) if (r, s) != (-2, 0) => {
                let axis = RotationAxis::new(dir.as_vec3(), Point3::origin());
                let tau = axis.dir.scaled_vec(frac!(s.abs()) / frac!(r.abs()));
                // dbg!(tau);
                let kind = RotationKind::new(true, r.abs() as usize);
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
                    PartialSymmOp::GenRotation(-2, 0) => Vector3::zero(),
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
    rotation: Option<PartialSymmOp>,
    /// Reflection component.
    reflection: Option<PartialSymmOp>,
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
    pub fn and(&mut self, op: PartialSymmOp) {
        if op.is_reflection() {
            self.reflection = self.reflection.as_ref().map_or_else(
                || Some(op.clone()),
                |r| r.partial_max(&op).or(Some(op.clone())),
            );
        } else {
            self.rotation = self.rotation.as_ref().map_or_else(
                || Some(op.clone()),
                |r| r.partial_max(&op).or(Some(op.clone())),
            );
        }
        // self.ops.push(op.clone());
    }

    /// Gets the rotation component, or the reflection, or the identity.
    pub fn first_op(&self) -> PartialSymmOp {
        self.rotation
            .or(self.reflection)
            .unwrap_or(PartialSymmOp::GenRotation(1, 0))
    }

    /// Gets the symmetry operations.
    pub fn ops(&self) -> Vec<PartialSymmOp> {
        vec![self.rotation.as_slice(), self.reflection.as_slice()].concat()
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
        reflection: Some(PartialSymmOp::GenRotation(-2, 0)),
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

impl SpaceGroupSetting {
    /// Generates a setting from a lattice and list of operations. Generates all operations from the
    /// provided operations. The identity operation is always assumed and does not need to be
    /// included.
    pub fn from_lattice_and_ops(
        lat_type: LatticeSystem,
        centering: HallCenteringType,
        ops: Vec<SymmOp>,
    ) -> Self {
        let mut setting = Self {
            lattice_type: lat_type,
            centering,
            linear_generators: ops,
            symmops: vec![],
        };

        setting.symmops = generate_elements(&setting);
        setting
    }

    pub fn op_list(&self) -> Vec<Block> {
        self.symmops
            .iter()
            .enumerate()
            .flat_map(|(i, op)| {
                let mut blocks = vec![Block::new_uint((i + 1) as u64), Block::new_text(": ")];
                blocks.append(&mut op.to_iso().modulo_unit_cell().components());
                blocks.push(Block::new_text("\n"));
                blocks
            })
            .collect()
    }

    pub fn full_hm_symbol(&self) -> (HallCenteringType, Vec<FullHMSymbolUnit>) {
        let dirs = self.lattice_type.all_symm_dirs();
        let mut syms = vec![];
        for _dir in &dirs {
            syms.push(FullHMSymbolUnit::default());
        }

        for op in self.symmops.clone() {
            let dir1 = op.symmetry_direction();
            let partial_op = PartialSymmOp::try_from_op(&op);
            for (i, dirlist) in (&dirs).into_iter().enumerate() {
                for dir2 in dirlist {
                    if let Some(((d1, d2), partial)) = dir1.zip(Some(dir2)).zip(partial_op.clone())
                    {
                        if d1 == *d2 {
                            syms[i].and(partial);
                        }
                    }
                }
            }
        }

        // special cases that don't fit the standard priority rules:
        // P-1 is P-1, not P1: only time -1 is shown
        if syms.len() == 1 && syms[0].first_op().rot_kind() == 1 {
            if self.symmops.len() > 1 {
                // not P1, so must be P-1
                syms[0].rotation = Some(PartialSymmOp::GenRotation(-1, 0));
            }
        };
        (self.centering, syms)
    }

    /// Short Hermann-Mauguin symbol. Removes the rotation component of most double symbols, except
    /// for the primary direction of monoclinic, tetragonal, and hexagonal groups. So P 6_3/m 2/m
    /// 2/c becomes P 6_3/m m c, and P 2/m 2/n 2_1/a becomes Pmna.
    pub fn short_hm_symbol(&self) -> (HallCenteringType, Vec<FullHMSymbolUnit>) {
        let full_hm = self.full_hm_symbol();
        let (center, all_syms) = full_hm.clone();
        let mut short_syms = all_syms.clone();

        // dbg!(&short_syms);

        // remove unnecessary 1s: anything except trigonal, because P321 and P312 are different, and
        // triclinic, because P1 is the only one!
        if (self.lattice_type != LatticeSystem::Hexagonal
            && short_syms[0].first_op().rot_kind() == 3)
            || (self.lattice_type != LatticeSystem::Triclinic)
        {
            short_syms.retain(|o| o.reflection.is_some() || o.first_op().rot_kind() != 1);
        }

        for i in 0..short_syms.len() {
            match (self.lattice_type, i) {
                (
                    LatticeSystem::Monoclinic
                    | LatticeSystem::Tetragonal
                    | LatticeSystem::Hexagonal,
                    0,
                ) => {
                    // keep the primary direction in this case
                    continue;
                }
                _ => {
                    // if a double unit, keep just the reflection
                    if short_syms[i].rotation.is_some() && short_syms[i].reflection.is_some() {
                        short_syms[i].rotation = None;
                    }
                }
            };
        }

        (center, short_syms)
    }
}

impl RenderBlocks for SpaceGroupSetting {
    fn components(&self) -> Vec<Block> {
        let (c, syms) = self.full_hm_symbol();
        let mut comps = c.components();
        for sym in syms {
            comps.push(Block::new_text(" ".into()));
            comps.extend(sym.components());
        }
        comps
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use nalgebra::Vector3;
    use pretty_assertions::{assert_eq, assert_str_eq};
    use proptest::proptest;

    use crate::{
        fract,
        isometry::Isometry,
        markup::{ASCII, DISPLAY, UNICODE},
        parsing::{hm_symbol, hm_unit},
    };

    use super::*;

    proptest! {
        #[test]
        fn test_rot_dir_decomp(
            r in -6_i8..=6_i8,
            s in 0_i8..=5_i8,
        ) {
            if r == 5 || r == -5 || r == 0 {
                return Ok(());
            }

            let s = (s % r.abs()).abs();

            // let r = if s != 0 {
            //     r.abs()
            // } else {
            //     r
            // };

            if (r, s) == (2, 1) {
                return Ok(());
            }

            let partial = PartialSymmOp::GenRotation(r, s);
            for ax in [Direction::new(Vector3::x()), Direction::new(Vector3::y()), Direction::new(Vector3::z())] {
                let op = partial.to_symmop_with_dir(ax);
                // assert_eq!(op.symmetry_direction(), Some(ax));
                assert_eq!(PartialSymmOp::try_from_op(&op), Some(partial));
            }
        }
    }

    #[test]
    fn test_pbcm_57_name() {
        let pbcm = SpaceGroupSetting::from_lattice_and_ops(
            LatticeSystem::Orthorhombic,
            HallCenteringType::P,
            vec![
                SymmOp::classify_affine("-x, -y, z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, y+1/2, -z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, -y, -z".parse().unwrap()).unwrap(),
            ],
        );

        println!("{}", UNICODE.render_to_string(&pbcm).as_str());
        assert_eq!(
            ASCII
                .render_to_string(&pbcm.short_hm_symbol().components())
                .as_str(),
            "Pbcm"
        );
        assert_eq!(ASCII.render_to_string(&pbcm).as_str(), "P 2/b 2_1/c 2_1/m");
        assert_eq!(UNICODE.render_to_string(&pbcm).as_str(), "P 2⁄b 2₁⁄c 2₁⁄m");
    }

    #[test]
    fn test_pbcm_57_residue() {
        let pbcm = SpaceGroupSetting::from_lattice_and_ops(
            LatticeSystem::Orthorhombic,
            HallCenteringType::P,
            vec![
                SymmOp::classify_affine("-x, -y, z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, y+1/2, -z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, -y, -z".parse().unwrap()).unwrap(),
            ],
        );

        let t1 = SymmOp::Translation(Vector3::new(frac!(-3 / 2), frac!(1 / 4), frac!(0)));
        let t2 = &SymmOp::Translation(Vector3::new(frac!(3 / 2), frac!(1 / 4), frac!(4)));

        dbg!(frac!(3 / 2), frac!(3 / 2) % frac!(1));

        assert!(
            pbcm.equiv(&t1, &t2),
            "{:?} != {:?}",
            pbcm.residue(&t1),
            pbcm.residue(&t2)
        );
    }

    #[test]
    fn test_pbcm_57_ops() {
        let pbcm = SpaceGroupSetting::from_lattice_and_ops(
            LatticeSystem::Orthorhombic,
            HallCenteringType::P,
            vec![
                SymmOp::classify_affine("-x, -y, z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, y+1/2, -z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, -y, -z".parse().unwrap()).unwrap(),
            ],
        );
        // for op in pbcm.clone().symmops {
        //     println!("{:?}", op.to_iso());
        //     println!("{:?}", op.to_iso().modulo_unit_cell());
        // }

        assert_str_eq!(
            ASCII.render_to_string(&pbcm.op_list()),
            "1: x, y, z
2: -x, -y, z +1/2
3: -x, y +1/2, -z +1/2
4: x, -y +1/2, -z
5: -x, -y, -z
6: x, y, -z +1/2
7: x, -y +1/2, z +1/2
8: -x, y +1/2, z
"
        );
    }

    #[test]
    fn test_pbcm_57_dirs() {
        let pbcm = SpaceGroupSetting::from_lattice_and_ops(
            LatticeSystem::Orthorhombic,
            HallCenteringType::P,
            vec![
                SymmOp::classify_affine("-x, -y, z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, y+1/2, -z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, -y, -z".parse().unwrap()).unwrap(),
            ],
        );

        for op in pbcm.symmops {
            println!(
                "{}\n{}\n{}\n{:?}",
                DISPLAY.render_to_string(&op.to_iso()),
                op.rotation_component().map_or_else(
                    || op
                        .translation_component()
                        .map(|t| format!("{:?}", t))
                        .unwrap_or("None".into()),
                    |t| format!("{:?} {:?}", t.axis.dir, t.kind)
                ),
                op.translation_component().unwrap_or(Vector3::zeros()),
                op.symmetry_direction().map(|d| d.v)
            );
            println!("\n---------------------------------------------\n");
        }
    }
}
