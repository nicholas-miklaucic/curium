//! Module to deal with units, using zero-cost compile-time checking to ensure dimensionality is
//! correct.

// TODO may be worth it to set the base unit to angstroms?
pub use uom::si::angle::{degree, radian};
pub use uom::si::f64::*;
pub use uom::si::length::angstrom;
pub use uom::si::Quantity;
pub use uom::unit;

unit! {
    system: uom::si;
    quantity: uom::si::electrical_conductivity;

    @megasiemens_per_meter: 1.0e-06; "MS/m", "megasiemens per meter", "megasiemens per meter";
}
