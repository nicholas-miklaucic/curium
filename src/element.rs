//! The base [`Element`] type. This code is mostly auto-generated, to allow for compile-time
//! generation of the large enum and its associated data without any runtime penalty.

// todo electrical_resistivity = 1 / electrical_conductivity
// X = pauling electronegativity
// wtf is atomic radius calculated if it's also empirical

use crate::units::*;
use uom::si::electrical_resistivity::ohm_meter;
use uom::si::length::picometer;
use uom::si::mass::dalton;
use uom::si::mass_density::kilogram_per_cubic_meter;
use uom::si::molar_energy::kilojoule_per_mole;
use uom::si::pressure::{gigapascal, megapascal};
use uom::si::ratio::ratio;
use uom::si::temperature_coefficient::per_kelvin;
use uom::si::thermal_conductivity::watt_per_meter_degree_celsius;
use uom::si::thermodynamic_temperature::kelvin;
use uom::si::time::year;
use uom::si::velocity::meter_per_second;

/// An element in the periodic table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    Hydrogen,
    Helium,
    Lithium,
    Beryllium,
    Boron,
    Carbon,
    Nitrogen,
    Oxygen,
    Fluorine,
    Neon,
    Sodium,
    Magnesium,
    Aluminium,
    Silicon,
    Phosphorus,
    Sulfur,
    Chlorine,
    Argon,
    Potassium,
    Calcium,
    Scandium,
    Titanium,
    Vanadium,
    Chromium,
    Manganese,
    Iron,
    Cobalt,
    Nickel,
    Copper,
    Zinc,
    Gallium,
    Germanium,
    Arsenic,
    Selenium,
    Bromine,
    Krypton,
    Rubidium,
    Strontium,
    Yttrium,
    Zirconium,
    Niobium,
    Molybdenum,
    Technetium,
    Ruthenium,
    Rhodium,
    Palladium,
    Silver,
    Cadmium,
    Indium,
    Tin,
    Antimony,
    Tellurium,
    Iodine,
    Xenon,
    Cesium,
    Barium,
    Lanthanum,
    Cerium,
    Praseodymium,
    Neodymium,
    Promethium,
    Samarium,
    Europium,
    Gadolinium,
    Terbium,
    Dysprosium,
    Holmium,
    Erbium,
    Thulium,
    Ytterbium,
    Lutetium,
    Hafnium,
    Tantalum,
    Tungsten,
    Rhenium,
    Osmium,
    Iridium,
    Platinum,
    Gold,
    Mercury,
    Thallium,
    Lead,
    Bismuth,
    Polonium,
    Astatine,
    Radon,
    Francium,
    Radium,
    Actinium,
    Thorium,
    Protactinium,
    Uranium,
    Neptunium,
    Plutonium,
    Americium,
    Curium,
    Berkelium,
    Californium,
    Einsteinium,
    Fermium,
    Mendelevium,
    Nobelium,
    Lawrencium,
    Rutherfordium,
    Dubnium,
    Seaborgium,
    Bohrium,
    Hassium,
    Meitnerium,
    Darmstadtium,
    Roentgenium,
    Copernicium,
    Nihonium,
    Flerovium,
    Moscovium,
    Livermorium,
    Tennessine,
    Oganesson,
    Ununennium, // from OUT_DIR/element_names.rs
                // you can't include a file to define enum variants :(
                // but I don't think new elements get defined that often
}

include!(concat!(env!("OUT_DIR"), "/element_data.rs"));

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use uom::si::pressure::megapascal;

    use super::*;

    #[test]
    fn test_element_names() {
        assert_eq!(Element::Hydrogen, Element::Hydrogen)
    }

    #[test]
    fn test_element_fields() {
        assert_eq!(Element::Carbon.atomic_number().unwrap(), 6);
        assert_abs_diff_eq!(
            Element::Boron
                .mineral_hardness()
                .unwrap()
                .get::<megapascal>(),
            9.3
        );
        assert_eq!(Element::Fluorine.half_life(), None);
        assert_abs_diff_eq!(
            Element::Fluorine
                .atomic_radius_empirical()
                .unwrap()
                .get::<picometer>(),
            50.
        );
        assert_abs_diff_eq!(
            Element::Zirconium
                .atomic_radius_calculated()
                .unwrap()
                .get::<picometer>(),
            206.
        );

        assert_abs_diff_eq!(
            Element::Berkelium.melting_point().unwrap().get::<kelvin>(),
            1259.
        );
    }
}
