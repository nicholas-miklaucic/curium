//! Describes space group symmetry in enough detail to perform the calculations necessary to derive
//! the information tabulated in ITA and other libraries. Unlike many other libraries, which require
//! storing all of this data per group, this information is intended for use only when needed. The
//! generic [`SpaceGroup`], a simple 230-element enum, is all that is needed to describe a space
//! group for serialization.

use crate::{
    algebra::{generate_elements, FiniteGroup, FinitelyGeneratedGroup, Group},
    lattice::{CenteringType, LatticeSystem},
    markup::{Block, RenderBlocks, ITA},
    symmop::{Direction, SymmOp},
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
        self.linear_generators.clone()
    }
}

impl FiniteGroup<SymmOp> for SpaceGroupSetting {
    type Elements = Vec<SymmOp>;

    fn elements(&self) -> Self::Elements {
        self.symmops.clone()
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
}

impl RenderBlocks for SpaceGroupSetting {
    fn components(&self) -> Vec<crate::markup::Block> {
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
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use nalgebra::Vector3;
    use pretty_assertions::assert_str_eq;

    use crate::{frac, isometry::Isometry, markup::ASCII};

    use super::*;

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
            ASCII.render_to_string(&pbcm),
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
}
