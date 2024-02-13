//! Describes space group symmetry in enough detail to perform the calculations necessary to derive
//! the information tabulated in ITA and other libraries. Unlike many other libraries, which require
//! storing all of this data per group, this information is intended for use only when needed. The
//! generic [`SpaceGroup`], a simple 230-element enum, is all that is needed to describe a space
//! group for serialization.

use std::str::FromStr;

use nalgebra::{Point3, Vector3};
use proptest::num;

use crate::{
    algebra::{generate_elements, FiniteGroup, FinitelyGeneratedGroup, Group},
    frac,
    frac::Frac,
    group_classes::CrystalSystem,
    lattice::{CenteringType, LatticeSystem},
    markup::{Block, RenderBlocks, ITA},
    symbols::{A_GLIDE, B_GLIDE, C_GLIDE, D_GLIDE, E_GLIDE, MIRROR, N_GLIDE},
    symmop::{Direction, Plane, RotationAxis, RotationKind, ScrewOrder, SymmOp},
};

/// A space group with a specific choice of symmetry directions. Essentially corresponds to a choice
/// of convention (e.g., rhombohedral or hexagonal axes) and then a full Hermann-Mauguin symbol. As
/// such, different `SpaceGroupSetting`s can represent the same space group number.
#[derive(Debug, Clone)]
pub struct SpaceGroupSetting {
    /// The coordinate system.
    lattice_type: LatticeSystem,
    /// The centering type of the lattice, defining the translational symmetry of the spacegroup.
    centering: CenteringType,
    /// The generator operations, not including pure translations.
    linear_generators: Vec<SymmOp>,
    /// The symmetry operations that define the group beyond translational symmetry. Should form a
    /// complete subgroup in the quotient group of the space group and the translational subgroup:
    /// i.e., when combining operations, the result should be a translated version of another
    /// operation.
    symmops: Vec<SymmOp>, // TODO this needs to properly account for centering type
}

/// Implements symmetry operation composition in the quotient group of the translational subgroup of
/// the group. More prosaically, this adds a mod 1 to everything: two points are equivalent if their
/// difference is integral.
impl Group<SymmOp> for SpaceGroupSetting {
    fn identity(&self) -> SymmOp {
        SymmOp::Identity
    }

    fn inv(&self, element: &SymmOp) -> SymmOp {
        element.inv()
    }

    fn compose(&self, a: &SymmOp, b: &SymmOp) -> SymmOp {
        self.residue(&a.compose(b))
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
        let mut gens = self.centering.centering_ops().unwrap();
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
                    let (b1, b2) = pl.basis();
                    let v1 = b1.as_vec3();
                    let v2 = b2.as_vec3();
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
            PartialSymmOp::GenRotation(_, 0) => 10,
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
        if let PartialSymmOp::GenRotation(r, s) = *self {
            match (r, s) {
                (1, 0) => SymmOp::Identity,
                (-1, 0) => SymmOp::Inversion(Point3::origin()),
                (r, s) => {
                    let axis = RotationAxis::new(dir.as_vec3(), Point3::origin());
                    let tau = axis.dir.scaled_vec(frac!(s.abs()) / frac!(r.abs()));
                    // dbg!(tau);
                    let kind = RotationKind::new(true, r.abs() as usize);
                    SymmOp::new_generalized_rotation(axis, kind, r.is_positive(), tau)
                }
            }
        } else {
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
                // TODO do we need all of the potential operations here? Can a symbol be mistakenly
                // excluded because one of its generators was redundant?
                PartialSymmOp::EGlide => v1.scale(f12),
                PartialSymmOp::NGlide => (v1 + v2).scale(f12),
                PartialSymmOp::DGlide => (v1 + v2).scale(f14),
                PartialSymmOp::GenRotation(_, _) => unreachable!(),
            };

            let plane = Plane::from_basis_and_origin(v1, v2, Point3::origin());
            SymmOp::new_generalized_reflection(plane, tau)
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
        centering: CenteringType,
        ops: Vec<SymmOp>,
    ) -> Self {
        if centering == CenteringType::SettingDependent {
            panic!("Can't do this yet");
        }

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

    pub fn full_hm_symbol(&self) -> (CenteringType, Vec<FullHMSymbolUnit>) {
        let dirs = self.lattice_type.symm_dirs();
        let mut syms = vec![];
        for _dir in &dirs {
            syms.push(FullHMSymbolUnit::default());
        }

        for op in self.symmops.clone() {
            let dir1 = op.symmetry_direction();
            let partial_op = PartialSymmOp::try_from_op(&op);
            for (i, dir2) in (&dirs).into_iter().enumerate() {
                if let Some(((d1, d2), partial)) = dir1.zip(*dir2).zip(partial_op.clone()) {
                    if d1 == d2 {
                        syms[i].and(partial);
                    }
                }
            }
        }

        (self.centering, syms)
    }

    /// Short Hermann-Mauguin symbol. Removes any symmetry operations that can be deduced from the
    /// others.
    pub fn short_hm_symbol(&self) -> (CenteringType, Vec<FullHMSymbolUnit>) {
        let full_hm = self.full_hm_symbol();
        let (center, all_syms) = full_hm.clone();
        let mut short_syms = all_syms.clone();
        let mut has_changed = true;
        while has_changed {
            println!("{:#?}", short_syms);
            has_changed = false;
            for i in 0..short_syms.len() {
                if let (Some(refl), Some(rot)) = (short_syms[i].reflection, short_syms[i].rotation)
                {
                    short_syms[i].rotation = None;
                    let new_op =
                        Self::from_hm_symbol(self.lattice_type, self.centering, &short_syms);
                    if new_op.full_hm_symbol() != full_hm {
                        // dbg!(new_op.full_hm_symbol());
                        short_syms[i].rotation = Some(rot);
                    } else {
                        has_changed = true;
                        break;
                    }

                    short_syms[i].reflection = None;
                    let new_op =
                        Self::from_hm_symbol(self.lattice_type, self.centering, &short_syms);
                    if new_op.full_hm_symbol() != full_hm {
                        // dbg!(new_op.full_hm_symbol());
                        short_syms[i].reflection = Some(refl);
                    } else {
                        has_changed = true;
                        break;
                    }
                }
            }
        }

        (center, short_syms)
    }

    /// Gets ops from lattice, centering, and full HM symbol.
    pub fn from_hm_symbol(
        lat_type: LatticeSystem,
        centering: CenteringType,
        hm_units: &[FullHMSymbolUnit],
    ) -> Self {
        let mut ops = vec![];
        for (dir, unit) in lat_type.symm_dirs().into_iter().zip(hm_units.into_iter()) {
            for partial in unit.ops() {
                ops.push(partial.to_symmop_with_dir(dir.unwrap()));
            }
        }

        Self::from_lattice_and_ops(lat_type, centering, ops)
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
        frac,
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
    fn test_from_hm_pnnn() {
        let pnnn = SpaceGroupSetting::from_lattice_and_ops(
            LatticeSystem::Orthorhombic,
            CenteringType::Primitive,
            vec![
                SymmOp::classify_affine("-x, -y, z".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, y, -z".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x+1/2, -y+1/2, -z+1/2".parse().unwrap()).unwrap(),
            ],
        );

        // let mut b2 = FullHMSymbolUnit::B.clone();
        // b2.and(PartialSymmOp::GenRotation(2, 0));

        let pnnn2 = SpaceGroupSetting::from_hm_symbol(
            LatticeSystem::Orthorhombic,
            CenteringType::Primitive,
            &[
                hm_unit("n").unwrap().1,
                hm_unit("n").unwrap().1,
                hm_unit("n").unwrap().1,
            ],
        );

        dbg!(&pnnn.linear_generators);

        println!(
            "\n{}\n{}\n{} {}",
            ITA.render_to_string(&pnnn.op_list()),
            ITA.render_to_string(&pnnn2.op_list()),
            ITA.render_to_string(&pnnn.full_hm_symbol()),
            ITA.render_to_string(&pnnn2.full_hm_symbol()),
        );

        assert_eq!(pnnn.full_hm_symbol(), pnnn2.full_hm_symbol());
    }

    #[test]
    fn test_from_hm_pbcm() {
        let pbcm = SpaceGroupSetting::from_lattice_and_ops(
            LatticeSystem::Orthorhombic,
            CenteringType::Primitive,
            vec![
                SymmOp::classify_affine("-x, -y, z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, y+1/2, -z+1/2".parse().unwrap()).unwrap(),
                SymmOp::classify_affine("-x, -y, -z".parse().unwrap()).unwrap(),
            ],
        );

        // let mut b2 = FullHMSymbolUnit::B.clone();
        // b2.and(PartialSymmOp::GenRotation(2, 0));

        let pbcm2 = SpaceGroupSetting::from_hm_symbol(
            LatticeSystem::Orthorhombic,
            CenteringType::Primitive,
            &[
                FullHMSymbolUnit::B,
                FullHMSymbolUnit::C,
                FullHMSymbolUnit::M,
            ],
        );

        assert_eq!(pbcm.full_hm_symbol(), pbcm2.full_hm_symbol());
    }

    #[test]
    fn test_pbcm_57_name() {
        let pbcm = SpaceGroupSetting::from_lattice_and_ops(
            LatticeSystem::Orthorhombic,
            CenteringType::Primitive,
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
            CenteringType::Primitive,
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
            CenteringType::Primitive,
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
            CenteringType::Primitive,
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

    #[test]
    fn test_parse_hm_47_59() {
        for name in
            "Pmmm Pnnn Pccn Pban Pmma Pnna Pmna Pcca Pbam Pccn Pbcm Pnnm Pmmn".split_whitespace()
        {
            let (o, (ctype, units)) = hm_symbol(name).unwrap();
            assert_eq!(o, "");
            let op = SpaceGroupSetting::from_hm_symbol(LatticeSystem::Orthorhombic, ctype, &units);

            assert_eq!(ASCII.render_to_string(&op.short_hm_symbol()), name);
        }
    }
}
