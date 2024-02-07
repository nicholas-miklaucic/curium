//! Types needed to classify the space groups into different taxa.

#[cfg(test)]
use proptest_derive::Arbitrary;

/// A crystal system, as defined in 2.1.1.1 (iii) of ITA. Combined with a centering type, which
/// defines the translation components, the result is a unique Bravais lattice.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CrystalSystem {
    Triclinic,
    Monoclinic,
    Orthorhombic,
    Tetragonal,
    Hexagonal,
    Trigonal,
    Cubic,
}

impl CrystalSystem {
    /// Gets the crystal family. This converts `Trigonal` to `Hexagonal` and leaves everything
    /// else unchanged.
    pub fn family(&self) -> Self {
        match *self {
            Self::Trigonal => Self::Hexagonal,
            other => other,
        }
    }
}

/// A crystal family. Describes groups of lattices with the same number of free parameters and
/// compatible point groups (in the sense of being subgroups of some supergroup.) There is only one
/// hexagonal family that comprises rhombohedral and hexagonal lattice systems.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(Arbitrary))]
pub enum CrystalFamily {
    Triclinic,
    Monoclinic,
    Orthorhombic,
    Tetragonal,
    Hexagonal,
    Cubic,
}
