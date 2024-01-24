//! Module to deal with units, using zero-cost compile-time checking to ensure dimensionality is
//! correct.

// TODO may be worth it to set the base unit to angstroms?
pub use uom::si::time::{year};
pub use uom::si::mass_concentration::{kilogram_per_cubic_meter};
pub use uom::si::molar_energy::{kilojoule_per_mole};
pub use uom::si::mass::{dalton};
pub use uom::si::thermodynamic_temperature::{kelvin};
pub use uom::si::pressure::{megapascal, gigapascal};
pub use uom::si::length::picometer;
pub use uom::si::electrical_resistivity::{ohm_meter};
pub use uom::si::temperature_coefficient::{per_kelvin};
pub use uom::si::thermal_conductivity::{watt_per_meter_degree_celsius};
pub use uom::si::velocity::{meter_per_second};
pub use uom::si::ratio::{ratio};
pub use uom::si::f32::*;
pub use uom::unit;

unit! {
    system: uom::si;
    quantity: uom::si::electrical_conductivity;

    @megasiemens_per_meter: 1.0e-06; "MS/m", "megasiemens per meter", "megasiemens per meter";
}