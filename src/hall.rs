//! Module for Hall symbol notation and display. Implements the specification as described [at
//! cci.lbl.gov](https://cci.lbl.gov/sginfo/hall_symbols.html).

use std::{default, f32::MIN, str::FromStr};

use nalgebra::{vector, Point3, Translation3, Vector, Vector3};
use simba::scalar::SupersetOf;
use thiserror::Error;

use crate::{
    frac,
    fract::Frac,
    lattice::LatticeSystem,
    markup::{Block, RenderBlocks},
    parsing::{hall_axis_symbol, hall_group},
    spacegroupdata::{RotOrder, SpaceGroupSetting},
    symbols::{LPAREN, MINUS_SIGN, RPAREN, SPACE},
    symmop::{
        Direction, Plane, RotationAxis, RotationDirectness, RotationKind, ScrewOrder, SymmOp,
    },
};

/// A Hall symbol description of a space group.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct HallGroupSymbol {
    /// Whether an inversion through the origin exists. Represented as a minus sign.
    pub has_inversion: bool,
    /// The centering type. More general than the Bravais centering types.
    pub centering: HallCenteringType,
    /// The non-translational generators.
    pub generators: Vec<HallOpSymbol>,
    /// The origin shift. Implicitly divides by 12.
    pub shift: (i8, i8, i8),
}

#[derive(Debug, Clone, Error)]
pub enum HallParseError {
    #[error("Invalid Hall symbol {0}")]
    InvalidHallSymbol(String),
}

impl FromStr for HallGroupSymbol {
    type Err = HallParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (o, hall) =
            hall_group(s).map_err(|_e| HallParseError::InvalidHallSymbol(s.to_string()))?;
        if o.is_empty() {
            Ok(hall)
        } else {
            Err(HallParseError::InvalidHallSymbol(s.to_string()))
        }
    }
}

impl HallGroupSymbol {
    /// Generates the group from the Hall symbol.
    pub fn generate_group(&self) -> SpaceGroupSetting {
        let mut linear_generators = vec![];

        let mut prev_n = 0;
        let mut prev_ax = None;

        let mut has_threefold = false;
        let mut max_order = 0;

        for (i, op) in self.generators.iter().enumerate() {
            let curr_n = op.order();

            let default_ax = if i == 0 {
                Some(HallAxisSymbol::Z)
            } else {
                match (i, prev_n, curr_n) {
                    // 1 or -1 doesn't matter
                    (_, _, 1) => Some(HallAxisSymbol::Z),
                    // second rotation is a or a - b depending on previous n for N = 2
                    // alternative notation is fine, but then axis must be explicit
                    (1, 2 | 4, 2) => Some(HallAxisSymbol::X),
                    (1, 3 | 6, 2) => Some(HallAxisSymbol::DoublePrime),
                    // third rotation is cubic diagonal if N = 3, otherwise must be explicit
                    (2, _, 3) => Some(HallAxisSymbol::Star),
                    _ => None,
                }
            };

            max_order = max_order.max(curr_n);
            if curr_n == 3 {
                has_threefold = true;
            }

            linear_generators.push(op.op(default_ax, prev_ax));
            prev_ax = op.axis.or(default_ax).map(|a| a.direction(prev_ax));
            prev_n = curr_n;
        }

        let lat_type = match (
            self.centering,
            max_order,
            has_threefold,
            linear_generators.len(),
        ) {
            (_, 1, false, 1) => LatticeSystem::Triclinic,
            (_, 2, false, 1) => LatticeSystem::Monoclinic,
            (_, 2, false, 2 | 3) => LatticeSystem::Orthorhombic,
            (_, 4, false, _) => LatticeSystem::Tetragonal,
            (HallCenteringType::P, 3, true, 1 | 2) => LatticeSystem::Rhombohedral,
            (HallCenteringType::R, 3, true, 1 | 2) => LatticeSystem::Hexagonal,
            (_, 6, _, _) => LatticeSystem::Hexagonal,
            (_, 2 | 3 | 4, true, 3) => LatticeSystem::Cubic,
            (a, b, c, d) => {
                dbg!(a, b, c, d);
                panic!("Uh oh!")
            }
        };

        if self.has_inversion {
            linear_generators.push(SymmOp::Inversion(Point3::origin()));
        }

        if self.shift != (0, 0, 0) {
            let f12 = frac!(12);
            let origin_shift = SymmOp::Translation(Vector3::new(
                frac!(self.shift.0) / f12,
                frac!(self.shift.1) / f12,
                frac!(self.shift.2) / f12,
            ));

            let origin_shift_inv = SymmOp::Translation(Vector3::new(
                frac!(self.shift.0) / -f12,
                frac!(self.shift.1) / -f12,
                frac!(self.shift.2) / -f12,
            ));

            linear_generators.iter_mut().for_each(|o| {
                let iso = origin_shift.to_iso() * o.to_iso() * origin_shift_inv.to_iso();
                println!("{} {}", iso, iso.modulo_unit_cell());
                *o = SymmOp::classify_affine(iso.modulo_unit_cell()).unwrap();
            });
        }

        for gen in &linear_generators {
            println!("gen: {}\n{}", gen, gen.to_iso());
        }

        SpaceGroupSetting::from_lattice_and_ops(lat_type, self.centering, linear_generators)
    }
}

impl RenderBlocks for HallGroupSymbol {
    fn components(&self) -> Vec<Block> {
        let mut blocks = if self.has_inversion {
            vec![MINUS_SIGN]
        } else {
            vec![]
        };

        blocks.extend(self.centering.components());

        for gen in &self.generators {
            blocks.push(SPACE);
            blocks.extend(gen.components());
        }

        if self.shift != (0, 0, 0) {
            blocks.extend(vec![
                SPACE,
                LPAREN,
                Block::new_int(self.shift.0 as i64),
                SPACE,
                Block::new_int(self.shift.1 as i64),
                SPACE,
                Block::new_int(self.shift.2 as i64),
                RPAREN,
            ]);
        }

        blocks
    }
}

/// The centering type of a space group: the translational generators. Supports unconventional
/// settings that the standard ITA symbols do not.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HallCenteringType {
    P,
    A,
    B,
    C,
    I,
    R,
    S,
    T,
    F,
}

impl HallCenteringType {
    /// Gets the letter for the symbol.
    pub fn letter(&self) -> char {
        match self {
            HallCenteringType::P => 'P',
            HallCenteringType::A => 'A',
            HallCenteringType::B => 'B',
            HallCenteringType::C => 'C',
            HallCenteringType::I => 'I',
            HallCenteringType::R => 'R',
            HallCenteringType::S => 'S',
            HallCenteringType::T => 'T',
            HallCenteringType::F => 'F',
        }
    }

    /// Gets the generators of the translational subgroup: the set of translation vectors that
    /// define the centering options. Includes the three basis vectors that shift by entire unit
    /// cells. Fails for `SettingDependent`.
    pub fn centering_ops(&self) -> Vec<SymmOp> {
        let mut translations = vec![Vector3::x(), Vector3::y(), Vector3::z()];

        let f0 = frac!(0);
        let f12 = frac!(1 / 2);
        let f13 = frac!(1 / 3);
        let f23 = frac!(2 / 3);

        let a = Vector3::new(f12, f0, f0);
        let b = Vector3::new(f0, f12, f0);
        let c = Vector3::new(f0, f0, f12);

        // Table 2.1.1.2 of ITA
        translations.extend(match self {
            Self::P => vec![],
            Self::I => vec![Vector3::new(f12, f12, f12)],
            Self::A => vec![a],
            Self::B => vec![b],
            Self::C => vec![c],
            Self::F => vec![a, b, c],
            Self::R => vec![Vector3::new(f23, f13, f13), Vector3::new(f13, f23, f23)],
            Self::T => vec![Vector3::new(f13, f23, f13), Vector3::new(f23, f13, f23)],
            Self::S => vec![Vector3::new(f13, f13, f23), Vector3::new(f23, f23, f13)],
        });

        translations
            .into_iter()
            .map(SymmOp::Translation)
            .collect::<Vec<SymmOp>>()
    }
}

impl RenderBlocks for HallCenteringType {
    fn components(&self) -> Vec<Block> {
        vec![Block::new_text(&format!("{}", self.letter()))]
    }
}

/// A rotation or rotoinversion with an optional screw. Unlike `SymmOp`'s version of the same, which
/// describes the sense of rotation, here we are only concerned with the group of rotations. Screw
/// rotoinversions are never used in Hall symbols, so they are omitted.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum RotationGroup {
    R1,
    Rm1,
    R2,
    Rm2,
    R21,
    R3,
    Rm3,
    R31,
    R32,
    R4,
    Rm4,
    R41,
    R42,
    R43,
    R6,
    Rm6,
    R61,
    R62,
    R63,
    R64,
    R65,
}

impl RotationGroup {
    /// Creates from rotation order, directness, and screw order. Fails if invalid.
    pub fn try_new(
        rot: RotOrder,
        directness: RotationDirectness,
        screw: ScrewOrder,
    ) -> Option<Self> {
        Some(match (rot, directness, screw) {
            (1, RotationDirectness::Proper, 0) => RotationGroup::R1,
            (1, RotationDirectness::Improper, 0) => RotationGroup::Rm1,
            (2, RotationDirectness::Proper, 0) => RotationGroup::R2,
            (2, RotationDirectness::Improper, 0) => RotationGroup::Rm2,
            (2, RotationDirectness::Proper, 1) => RotationGroup::R21,
            (3, RotationDirectness::Proper, 0) => RotationGroup::R3,
            (3, RotationDirectness::Improper, 0) => RotationGroup::Rm3,
            (3, RotationDirectness::Proper, 1) => RotationGroup::R31,
            (3, RotationDirectness::Proper, 2) => RotationGroup::R32,
            (4, RotationDirectness::Proper, 0) => RotationGroup::R4,
            (4, RotationDirectness::Improper, 0) => RotationGroup::Rm4,
            (4, RotationDirectness::Proper, 1) => RotationGroup::R41,
            (4, RotationDirectness::Proper, 2) => RotationGroup::R42,
            (4, RotationDirectness::Proper, 3) => RotationGroup::R43,
            (6, RotationDirectness::Proper, 0) => RotationGroup::R6,
            (6, RotationDirectness::Improper, 0) => RotationGroup::Rm6,
            (6, RotationDirectness::Proper, 1) => RotationGroup::R61,
            (6, RotationDirectness::Proper, 2) => RotationGroup::R62,
            (6, RotationDirectness::Proper, 3) => RotationGroup::R63,
            (6, RotationDirectness::Proper, 4) => RotationGroup::R64,
            (6, RotationDirectness::Proper, 5) => RotationGroup::R65,
            _ => {
                return None;
            }
        })
    }
    /// Gets the rotation order, directness, and screw order of the rotation group.
    pub fn order_directness_screw(&self) -> (RotOrder, RotationDirectness, ScrewOrder) {
        match *self {
            RotationGroup::R1 => (1, RotationDirectness::Proper, 0),
            RotationGroup::Rm1 => (1, RotationDirectness::Improper, 0),
            RotationGroup::R2 => (2, RotationDirectness::Proper, 0),
            RotationGroup::Rm2 => (2, RotationDirectness::Improper, 0),
            RotationGroup::R21 => (2, RotationDirectness::Proper, 1),
            RotationGroup::R3 => (3, RotationDirectness::Proper, 0),
            RotationGroup::Rm3 => (3, RotationDirectness::Improper, 0),
            RotationGroup::R31 => (3, RotationDirectness::Proper, 1),
            RotationGroup::R32 => (3, RotationDirectness::Proper, 2),
            RotationGroup::R4 => (4, RotationDirectness::Proper, 0),
            RotationGroup::Rm4 => (4, RotationDirectness::Improper, 0),
            RotationGroup::R41 => (4, RotationDirectness::Proper, 1),
            RotationGroup::R42 => (4, RotationDirectness::Proper, 2),
            RotationGroup::R43 => (4, RotationDirectness::Proper, 3),
            RotationGroup::R6 => (6, RotationDirectness::Proper, 0),
            RotationGroup::Rm6 => (6, RotationDirectness::Improper, 0),
            RotationGroup::R61 => (6, RotationDirectness::Proper, 1),
            RotationGroup::R62 => (6, RotationDirectness::Proper, 2),
            RotationGroup::R63 => (6, RotationDirectness::Proper, 3),
            RotationGroup::R64 => (6, RotationDirectness::Proper, 4),
            RotationGroup::R65 => (6, RotationDirectness::Proper, 5),
        }
    }
}

/// A Hall symbol for a symmetry operation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct HallOpSymbol {
    /// The rotation component.
    pub rotation: RotationGroup,
    /// An axis specification, if required.
    pub axis: Option<HallAxisSymbol>,
    /// The translation part of the symmetry operation.
    pub translation: Vec<HallTranslationSymbol>,
}

impl HallOpSymbol {
    pub fn order(&self) -> RotOrder {
        self.rotation.order_directness_screw().0
    }

    pub fn op(&self, default_axis: Option<HallAxisSymbol>, prev_axis: Option<Direction>) -> SymmOp {
        let (ord, proper, screw) = self.rotation.order_directness_screw();
        let axis = self.axis.or(default_axis).unwrap();
        let (tx, ty, tz) = self
            .translation
            .iter()
            .map(|t| t.vector())
            .fold((frac!(0), frac!(0), frac!(0)), |a, b| {
                (a.0 + b.0, a.1 + b.1, a.2 + b.2)
            });
        let tau = vector![tx, ty, tz];

        let dir = axis.direction(prev_axis);
        let (b1, b2) = dir.plane_basis();
        let (v1, v2) = (b1.as_vec3(), b2.as_vec3());

        let screw_tau = dir.scaled_vec(frac!(screw.abs()) / frac!(ord.abs()));

        let rot = match self.rotation {
            RotationGroup::R1 => SymmOp::Identity,
            RotationGroup::Rm1 => SymmOp::Inversion(Point3::origin()),
            RotationGroup::Rm2 => SymmOp::new_generalized_reflection(
                Plane::from_basis_and_origin(v1, v2, Point3::origin()),
                tau,
            ),
            _ => SymmOp::new_generalized_rotation(
                RotationAxis::new(dir.as_vec3(), Point3::origin()),
                RotationKind::new(true, ord as usize),
                proper == RotationDirectness::Proper,
                screw_tau,
            ),
        };

        let tau_op = SymmOp::Translation(tau);
        tau_op.compose(&rot).compose(&tau_op.inv())
    }
}

impl RenderBlocks for HallOpSymbol {
    fn components(&self) -> Vec<Block> {
        let (o, d, s) = self.rotation.order_directness_screw();
        let mut blocks = vec![Block::new_int((o * d.det_sign() as i8) as i64)];
        if s != 0 {
            blocks.push(Block::new_uint(s.rem_euclid(o) as u64));
        }

        blocks.extend(self.axis.map(|x| x.components()).unwrap_or_default());

        for tau in &self.translation {
            blocks.extend(tau.components());
        }

        blocks
    }
}

/// An axis specification for a symmetry operation.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HallAxisSymbol {
    X,
    Y,
    Z,
    Prime,
    DoublePrime,
    Star,
}

impl HallAxisSymbol {
    /// Gets the letter for the symbol.
    pub fn letter(&self) -> char {
        match *self {
            HallAxisSymbol::X => 'x',
            HallAxisSymbol::Y => 'y',
            HallAxisSymbol::Z => 'z',
            HallAxisSymbol::Prime => '\'',
            HallAxisSymbol::DoublePrime => '"',
            HallAxisSymbol::Star => '*',
        }
    }

    /// Gets the axis direction. Because the face diagonals depend on the preceding axis, this takes
    /// in an `Option<Direction>` specifying that axis. If `None`, the default of `c` is used.
    pub fn direction(&self, prev_axis: Option<Direction>) -> Direction {
        let (b1, b2) = prev_axis
            .unwrap_or(Direction::new(Vector3::z()))
            .plane_basis();

        let (v1, v2) = (b1.as_vec3(), b2.as_vec3());
        match *self {
            HallAxisSymbol::X => Direction::new(Vector3::x()),
            HallAxisSymbol::Y => Direction::new(Vector3::y()),
            HallAxisSymbol::Z => Direction::new(Vector3::z()),
            HallAxisSymbol::Prime => Direction::new(v1 + v2),
            HallAxisSymbol::DoublePrime => Direction::new(v1 - v2),
            HallAxisSymbol::Star => Direction::new(Vector3::x() + Vector3::y() + Vector3::z()),
        }
    }
}

impl RenderBlocks for HallAxisSymbol {
    fn components(&self) -> Vec<Block> {
        vec![Block::new_text(&format!("{}", self.letter()))]
    }
}

/// A Hall symbol for a translation.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum HallTranslationSymbol {
    A,
    B,
    C,
    N,
    U,
    V,
    W,
    D,
}

impl HallTranslationSymbol {
    /// Gets the letter for the symbol.
    pub fn letter(&self) -> char {
        match self {
            HallTranslationSymbol::A => 'a',
            HallTranslationSymbol::B => 'b',
            HallTranslationSymbol::C => 'c',
            HallTranslationSymbol::N => 'n',
            HallTranslationSymbol::U => 'u',
            HallTranslationSymbol::V => 'v',
            HallTranslationSymbol::W => 'w',
            HallTranslationSymbol::D => 'd',
        }
    }

    /// Gets the translation vector.
    pub fn vector(&self) -> (Frac, Frac, Frac) {
        let f0 = frac!(0);
        let f14 = frac!(1 / 4);
        let f12 = frac!(1 / 2);
        match *self {
            HallTranslationSymbol::A => (f12, f0, f0),
            HallTranslationSymbol::B => (f0, f12, f0),
            HallTranslationSymbol::C => (f0, f0, f12),
            HallTranslationSymbol::N => (f12, f12, f12),
            HallTranslationSymbol::U => (f14, f0, f0),
            HallTranslationSymbol::V => (f0, f14, f0),
            HallTranslationSymbol::W => (f0, f0, f14),
            HallTranslationSymbol::D => (f14, f14, f14),
        }
    }
}

impl RenderBlocks for HallTranslationSymbol {
    fn components(&self) -> Vec<Block> {
        vec![Block::new_text(&format!("{}", self.letter()))]
    }
}

#[cfg(test)]
mod tests {
    use crate::markup::{ASCII, ITA};

    use super::*;

    #[test]
    fn test_hall_parse() {
        let cases = [
            ("P 2 2 3", "P23"),
            ("P 6", "P6"),
            ("F 2 -2d", "Fdd2"),
            ("I -4 -2", "I-4m2"),
            // ("P 61 2 (0 0 -1)", "P6522"),
            // ("-I 4bd 2c 3", "Ia-3d"),
        ];

        for (hall, hm) in cases {
            let group: HallGroupSymbol = hall.parse().unwrap();
            let setting = group.generate_group();
            // dbg!(&setting);
            println!(
                "{:?}\n{}",
                setting.lattice_type,
                ITA.render_to_string(&setting.op_list())
            );
            assert_eq!(&ASCII.render_to_string(&setting.short_hm_symbol()), hm);
        }
    }
}
