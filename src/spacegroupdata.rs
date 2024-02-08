//! Describes space group symmetry in enough detail to perform the calculations necessary to derive
//! the information tabulated in ITA and other libraries. Unlike many other libraries, which require
//! storing all of this data per group, this information is intended for use only when needed. The
//! generic [`SpaceGroup`], a simple 230-element enum, is all that is needed to describe a space
//! group for serialization.

use nalgebra::{Point3, Vector3};

use crate::{
    algebra::{generate_elements, FiniteGroup, FinitelyGeneratedGroup, Group},
    frac,
    frac::Frac,
    group_classes::CrystalSystem,
    lattice::{CenteringType, LatticeSystem},
    markup::{Block, RenderBlocks, ITA},
    symbols::{A_GLIDE, B_GLIDE, C_GLIDE, D_GLIDE, E_GLIDE, MIRROR, N_GLIDE},
    symmop::{Direction, ScrewOrder, SymmOp},
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
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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
                vec![
                    Block::new_int(r as i64),
                    Block::new_text("_"),
                    Block::new_uint((s % r).abs() as u64),
                ]
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
pub struct FullHMSymbol {
    /// Rotation component.
    rotation: Option<PartialSymmOp>,
    /// Reflection component.
    reflection: Option<PartialSymmOp>,
    pub ops: Vec<PartialSymmOp>,
}

impl FullHMSymbol {
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
        self.ops.push(op.clone());
    }
}

impl RenderBlocks for FullHMSymbol {
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

    pub fn full_hm_symbol(&self) -> (CenteringType, Vec<FullHMSymbol>) {
        let dirs = self.lattice_type.symm_dirs();
        let mut syms = vec![];
        for _dir in &dirs {
            syms.push(FullHMSymbol::default());
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
    use pretty_assertions::assert_str_eq;

    use crate::{
        frac,
        isometry::Isometry,
        markup::{ASCII, DISPLAY, UNICODE},
    };

    use super::*;

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

        // dbg!(pbcm.full_hm_symbol().1);
        assert_eq!(ASCII.render_to_string(&pbcm).as_str(), "P 2/b 2_1/c 2_1/m");
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
}
