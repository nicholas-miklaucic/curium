//! Describes space group symmetry in enough detail to perform the calculations necessary to derive
//! the information tabulated in ITA and other libraries. Unlike many other libraries, which require
//! storing all of this data per group, this information is intended for use only when needed. The
//! generic [`SpaceGroup`], a simple 230-element enum, is all that is needed to describe a space
//! group for serialization.

use std::{fmt::Display, str::FromStr};

use nalgebra::{Point3, Translation3, Vector3};
use num_traits::Zero;

use crate::{
    algebra::{generate_elements, FiniteGroup, FinitelyGeneratedGroup, Group},
    constants::{HALL_SYMBOLS, SPACE_GROUP_SYMBOLS},
    frac,
    fract::Frac,
    geometry::{Direction, Plane, RotationAxis},
    group_classes::CrystalSystem,
    hall::{HallCenteringType, HallGroupSymbol},
    hermann_mauguin::{FullHMSymbolUnit, PartialSymmOp},
    isometry::Isometry,
    lattice::{CenteringType, LatticeSystem},
    markup::{Block, RenderBlocks, ITA},
    symbols::{A_GLIDE, B_GLIDE, C_GLIDE, D_GLIDE, E_GLIDE, LPAREN, MIRROR, N_GLIDE, RPAREN},
    symmop::{RotationKind, ScrewOrder, SymmOp},
};

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
    pub symmops: Vec<SymmOp>,
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
        let is_hex = self.is_hex();
        // println!("{a} * {b} = ");
        // println!(
        //     " {} * {} = ",
        //     ITA.render_to_string(&a.to_iso(is_hex)),
        //     ITA.render_to_string(&b.to_iso(is_hex))
        // );
        // println!(
        //     "---------------\n {}\n-------------",
        //     ITA.render_to_string(&(a.to_iso(is_hex) * b.to_iso(is_hex)).modulo_unit_cell())
        // );

        // We have to reduce the isometry modulo a unit cell *before* we convert to a SymmOp. The
        // reason is that our Fracs can't handle all of the potential SymmOps in the total space of
        // operations, but we can represent all of the SymmOps we actually need. For example,
        // combining a rotation y, z, x with t(0, 0, 1) seems harmless, but the resulting axis is
        // quite weird.
        let el_res =
            SymmOp::classify_affine((a.to_iso(is_hex) * b.to_iso(is_hex)).modulo_unit_cell());
        let el = match el_res {
            Ok(el) => el,
            Err(_e) => {
                println!("{a} * {b} = ");
                println!(
                    " {} * {} = ",
                    ITA.render_to_string(&a.to_iso(is_hex)),
                    ITA.render_to_string(&b.to_iso(is_hex))
                );
                println!(
                    "{}",
                    ITA.render_to_string(&(a.to_iso(is_hex) * b.to_iso(is_hex)).modulo_unit_cell())
                );
                panic!();
            }
        };
        // println!("el: {el}");
        assert_eq!(SymmOp::classify_affine(el.to_iso(is_hex)).unwrap(), el);
        self.residue(&el)
    }

    fn equiv(&self, a: &SymmOp, b: &SymmOp) -> bool {
        self.residue(a) == self.residue(b)
    }

    fn residue(&self, el: &SymmOp) -> SymmOp {
        el.modulo_unit_cell()
    }
}

impl FinitelyGeneratedGroup<SymmOp> for SpaceGroupSetting {
    type Generators = Vec<SymmOp>;

    fn generators(&self) -> Self::Generators {
        let centerings = self.centering.centering_ops();
        let mut gens = vec![];
        gens.extend(self.linear_generators.clone());
        gens.push(SymmOp::Translation(Vector3::x()));
        gens.push(SymmOp::Translation(Vector3::y()));
        gens.push(SymmOp::Translation(Vector3::z()));
        gens.extend(centerings);
        gens
    }
}

impl FiniteGroup<SymmOp> for SpaceGroupSetting {
    type Elements = Vec<SymmOp>;

    fn elements(&self) -> Self::Elements {
        self.symmops.clone()
    }
}

impl SpaceGroupSetting {
    pub fn from_hm_symbol_lookup(symbol: &str) -> Self {
        let sym = symbol.replace(" ", "").replace("_", "");
        Self::from_number(
            SPACE_GROUP_SYMBOLS
                .into_iter()
                .position(|s| s == sym)
                .unwrap(),
        )
    }

    /// Gets all of the `SymmOp`s, including the centering translations.
    pub fn all_symmops(&self) -> Vec<SymmOp> {
        let mut ops = self.symmops.clone();
        for tau in self.centering.centering_ops() {
            if !self.contains_equiv(&self.symmops, &tau) {
                for op in &self.symmops {
                    ops.push(tau.compose(op));
                }
            }
        }
        ops
    }

    /// Whether the group has a symmop matching the given one.
    pub fn contains_op(&self, op: &SymmOp) -> bool {
        self.contains_equiv(&self.all_symmops(), op)
    }

    /// Gets the translation vector that corresponds to this SymmOp.
    pub fn centering_component(&self, op: &SymmOp) -> Option<Vector3<Frac>> {
        let mut centers = self.centering.centering_ops();
        centers.push(SymmOp::Translation(Vector3::zeros()));

        centers
            .into_iter()
            .find(|c| self.contains_op(&c.inv().compose(op)))
            .map(|t| t.translation_component().unwrap())
    }

    /// Gets just the `SymmOps` without the centering translation.
    pub fn uncentered_symmops(&self) -> Vec<SymmOp> {
        self.all_symmops()
            .into_iter()
            .filter(|op| self.centering_component(op).is_some_and(|v| v.is_zero()))
            .collect()
    }

    /// Gets the space group corresponding to the number given. Panics if the
    /// number is incorrect.
    pub fn from_number(group_num: usize) -> Self {
        for (hall, grp) in HALL_SYMBOLS {
            if grp == group_num {
                let group: HallGroupSymbol = hall.parse().unwrap();
                return group.generate_group();
            }
        }

        panic!("No group found for {}!", group_num)
    }

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

    pub fn is_hex(&self) -> bool {
        // TODO this needs to be refactored to support rhombohedral coordinates as well
        self.lattice_type == LatticeSystem::Hexagonal
    }

    /// Formats a list of all of the ops, giving both the Jones notation and the geometric meaning.
    /// Gets all of the isometries in the space group.
    pub fn op_isometries(&self) -> Vec<Isometry> {
        self.all_symmops()
            .iter()
            .map(|s| s.to_iso(self.is_hex()))
            .collect()
    }

    /// Formats a list of all of the ops.
    pub fn op_list(&self) -> Vec<Block> {
        self.symmops
            .iter()
            .enumerate()
            .flat_map(|(i, op)| {
                let mut blocks = vec![Block::new_uint((i + 1) as u64), Block::new_text(": ")];
                blocks.extend(op.to_iso(self.is_hex()).modulo_unit_cell().components());
                blocks.push(Block::new_text("\t|\t"));
                blocks.extend(op.components());
                blocks.push(Block::new_text("\n"));
                blocks
            })
            .collect()
    }

    /// Formats a list of all of the ops, in Jones faithful notation.
    pub fn op_list_jones(&self) -> Vec<Block> {
        self.symmops
            .iter()
            .enumerate()
            .flat_map(|(i, op)| {
                let mut blocks = vec![Block::new_uint((i + 1) as u64), Block::new_text(": ")];
                blocks.append(&mut op.to_iso(self.is_hex()).modulo_unit_cell().components());
                blocks.push(Block::new_text("\n"));
                blocks
            })
            .collect()
    }

    /// Formats a list of all of the ops as their geometric meaning.
    pub fn op_list_geometric(&self) -> Vec<Block> {
        self.symmops
            .iter()
            .enumerate()
            .flat_map(|(i, op)| {
                let mut blocks = vec![LPAREN, Block::new_uint((i + 1) as u64), RPAREN];
                blocks.extend(op.components());
                blocks.push(Block::new_text("\n"));
                blocks
            })
            .collect()
    }

    pub fn full_hm_symbol(&self) -> (HallCenteringType, Vec<FullHMSymbolUnit>) {
        let dirs = self.lattice_type.all_symm_dirs();
        let use_e_glide = matches!(self.centering, HallCenteringType::A | HallCenteringType::C);
        let mut syms = vec![];
        for _dir in &dirs {
            syms.push(FullHMSymbolUnit::default());
        }

        for op in self.uncentered_symmops() {
            let dir1 = op.symmetry_direction();
            let partial_op = PartialSymmOp::try_from_op(&op);
            for (i, dirlist) in dirs.iter().enumerate() {
                for dir2 in dirlist {
                    if let Some(((d1, d2), partial)) = dir1.zip(Some(dir2)).zip(partial_op) {
                        if d1 == *d2 {
                            syms[i].and(partial, use_e_glide);
                        }
                    }
                }
            }
        }

        // special cases that don't fit the standard priority rules:
        // P-1 is P-1, not P1: only time -1 is shown
        if syms.len() == 1 && syms[0].first_op().rot_kind() == 1 && self.symmops.len() > 1 {
            // not P1, so must be P-1
            syms[0].rotation = Some(PartialSymmOp::GenRotation(-1, frac!(0)));
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

        for (i, sym) in short_syms.iter_mut().enumerate() {
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
                    if sym.rotation.is_some() && sym.reflection.is_some() {
                        sym.rotation = None;
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
            comps.push(Block::new_text(" "));
            comps.extend(sym.components());
        }
        comps
    }
}

impl Display for SpaceGroupSetting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ITA.render_to_string(&self.op_list()))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, str::FromStr};

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

    #[test]
    fn test_centering_ops() {
        for grp_sym in ["I4_1/acd"] {
            let mut counts: HashMap<Vector3<Frac>, usize> = HashMap::new();
            let group = SpaceGroupSetting::from_hm_symbol_lookup(grp_sym);
            for op in group.all_symmops() {
                let comp = group
                    .centering_component(&op)
                    .unwrap_or_else(|| panic!("{op} {group}"));
                *counts.entry(comp).or_default() += 1;
            }

            println!("{counts:?}");
            assert_eq!(counts.values().max(), counts.values().min());
        }
    }

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

            let partial = PartialSymmOp::GenRotation(r, frac!(s) / frac!(r));
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
        let t2 = SymmOp::Translation(Vector3::new(frac!(3 / 2), frac!(1 / 4), frac!(4)));

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
        //     println!("{:?}", op.to_iso(false));
        //     println!("{:?}", op.to_iso(false).modulo_unit_cell());
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
                "{}\n{}\n{}",
                DISPLAY.render_to_string(&op.to_iso(false)),
                op.rotation_component().map_or_else(
                    || op
                        .translation_component()
                        .map(|t| format!("{:?}", t))
                        .unwrap_or("None".into()),
                    |t| format!("{:?} {:?}", t.axis.dir(), t.kind)
                ),
                op.translation_component().unwrap_or(Vector3::zeros()),
            );
            println!("\n---------------------------------------------\n");
        }
    }
}
