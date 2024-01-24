// Generates the element data Rust code from the CSV file.

// The data is due to https://github.com/sweaver2112/periodic-table-data. Periodic table data is
// public, but there may be errors, and if there are any licensing issues or mistakes please file an
// issue at https://github.com/nicholas-miklaucic/curium. 

use std::collections::HashMap;
use std::fmt::Display;
use std::{env, fs::File};
use std::io::BufReader;
use std::path::Path;
use serde::de::Error as DeError;
use serde_json::{json, Error, Value};
use std::error::Error as StdError;


use phf::phf_map;

fn display_opt<T: Display>(o: Option<T>) -> String {
    match o {
        Some(t) => format!("Some({})", t),
        None => "None".to_string()
    }
}

fn floatify<T: Display>(t: T) -> String {
    let mut out = format!("{:.6}", t);
    if !out.contains('.') && !out.contains("e-") {
        out.push('.');
    }
    out
}

/// Map to the name to use in code and the unit.
static COLUMNS: phf::Map<&'static str, (&'static str, &'static str, &'static str)> = phf_map! {
    "atomic_mass" => ("atomic_mass", "Mass", "dalton"),
    "boiling_point" => ("boiling_point", "ThermodynamicTemperature", "kelvin"),
    "melting_point" => ("melting_point", "ThermodynamicTemperature", "kelvin"),
    "conductivity/thermal" => ("thermal_conductivity", "ThermalConductivity", "watt_per_meter_degree_celsius"),
    "conductivity/electric" => ("electric_conductivity" , "ElectricalConductivity", "megasiemens_per_meter"),
    "mendeleev_no" => ("mendeleev_no", "u8", "u8"),
    "density/stp" => ("stp_density", "MassDensity", "kilogram_per_cubic_meter"),
    "electron_affinity" => ("electron_affinity", "MolarEnergy", "kilojoule_per_mole"),
    "electronegativity_pauling" => ("pauling_electronegativity", "MolarEnergy", "kilojoule_per_mole"),
    // "energy_levels" => ("energy_levels", "Vec<u8>", "u8"),
    // whyyyyy
    "half-life" => ("half_life", "Time", "year"),
    "hardness/vickers" => ("vickers_hardness", "Pressure", "megapascal"),
    "hardness/brinell" => ("brinell_hardness", "Pressure", "megapascal"),
    "hardness/mohs" => ("mineral_hardness", "Pressure", "megapascal"),
    "modulus/bulk" => ("bulk_modulus", "Pressure", "gigapascal"),
    "radius/calculated" => ("atomic_radius_calculated", "Length", "picometer"),
    "radius/vanderwalls" => ("van_der_waals_radius", "Length", "picometer"),
    "radius/empirical" => ("atomic_radius_empirical", "Length", "picometer"),
    "resistivity" => ("electrical_resistivity", "ElectricalResistivity", "ohm_meter"),
    "thermal_expansion" => ("coefficient_of_linear_thermal_expansion", "TemperatureCoefficient", "per_kelvin"),
    "speed_of_sound" => ("velocity_of_sound", "Velocity", "meter_per_second"),
    "ionization_energies" => ("ionization_energies", "Vec<MolarEnergy>", "kilojoule_per_mole"),
    "atomic_number" => ("atomic_number", "u8", "u8"),
    "poisson_ratio" => ("poisson_ratio", "Ratio", "ratio"),    
};

fn main() -> Result<(), Box<dyn StdError>> {    
    let out_dir = env::var_os("OUT_DIR").ok_or(std::env::VarError::NotPresent)?;    
    let mut ptable_path = env::current_dir()?;
    ptable_path.push("data/ptable.json");
    let data_file = File::open(ptable_path)?;
    let reader = BufReader::new(data_file);
    let data: Value = serde_json::from_reader(reader)?;

    let ptable = match data {
        Value::Object(map) => {
            match map.get("pTable").ok_or(Error::missing_field("pTable")) {
                Ok(Value::Array(pdata)) => Ok(pdata.clone()),
                Ok(_) => Err(Error::missing_field("pTable")),
                Err(e) => Err(e),
            }
        },
        _ => Err(Error::missing_field("pTable")),
    }?;
    
    let name_templ = r#"
/// An element in the periodic table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    //$element_names
}"#;

    let element_names: Option<Vec<&Value>> = ptable.iter().map(|e| e.get("name")).collect();
    let element_names = element_names.ok_or(Error::missing_field("name"))?;
    let element_names: Vec<&str> = element_names.iter().map(|v| v.as_str().unwrap()).collect();    

    let name_templ = name_templ.replace("//$element_names", element_names.join(",\n    ").as_str());

    let dest_path = Path::new(&out_dir).join("element_names.rs");
    std::fs::write(dest_path, name_templ).unwrap();
    
    fn make_function(data_name: &str, field_name: &str, type_name: &str, 
        unit_name: &str, values: &Vec<Value>) -> String {
        let mut match_map = HashMap::new();
        for val in values {         
/*             match val.get("name") {
                Some(e) => {if !e.is_array() {
                    dbg!(e);
                }},
                None => {dbg!(val);}
            }; */
            let data_name_parts = data_name.split('/').collect::<Vec<_>>();
            let (data_name, inner_val) =  match data_name_parts.as_slice() {
                    [outer, inner] => (inner, match val.get(outer) {
                        Some(nested_val) => nested_val.clone(),
                        None => {
                            if inner == &"mohs" && val.get("name") == Some(&Value::String("Boron".into())) {
                                dbg!(&val, outer);
                                panic!("uh");
                            }
                            json!({})
                        }
                    }),
                    [single] => (single, val.clone()),
                    _ => {
                        dbg!(val, data_name);
                        panic!("uh");
                    }
                };
            let name = val.get("name").unwrap().as_str().unwrap();
            if type_name.starts_with("Vec<") {
                let inner_type_name = type_name.trim_start_matches("Vec<").trim_end_matches('>');
                let mut data = vec![];
                
                match inner_val.get(data_name).map(|f| f.is_array()) {
                    Some(true) => {},
                    Some(false) => {dbg!(data_name, inner_val.get(data_name).expect("crap"));} 
                    None => {dbg!(&inner_val, data_name);},
                }
                for element in inner_val.get(data_name).unwrap_or(&Value::Array(vec![])).as_array().expect("uh") {
                    match inner_type_name.chars().next().unwrap() {
                        'u' | 'i' => {
                            // numeric type, string conversion is fine and type inference will
                            // handle the rest
                            data.push(format!("{}", element));
                        },
                        'f' => {
                            // have to add dot if not there
                            data.push(floatify(element));
                        },
                        _ => {
                            // need to construct type ourselves
                            let f_el = floatify(element);
                            data.push(format!("{inner_type_name}::new::<{unit_name}>({f_el})"));
                        }
                    }
                }

                match_map.insert(name, format!("vec![{}]", data.join(",")));
            } else {
                let val_constructor = match type_name.chars().next().unwrap() {
                    'u' | 'i' => {
                        // numeric type, string conversion is fine and type inference will
                        // handle the rest
                        display_opt(inner_val.get(data_name))
                    },
                    'f' => {
                        // have to add dot if not there
                        display_opt(inner_val.get(data_name).map(floatify))
                    },
                    _ => {
                        // need to construct type ourselves
                        let f_val = inner_val.get(data_name).map(|v| {
                            let fv = floatify(v);
                            format!("{type_name}::new::<{unit_name}>({fv})")});
                        display_opt(f_val)
                    }
                };
                match_map.insert(name, val_constructor);
            }
        }
        let func_body: Vec<String> = match_map.into_iter().map(|(k, v)| {
            format!("Self::{} => {}", k, v)
        }).collect();
        let func_body = func_body.join(",\n");
        let out_type = if func_body.contains("Some") || func_body.contains("None") {
            format!("Option<{type_name}>")
        } else {
            type_name.to_string()
        };
        format!(r#"
/// Get this constant for the element. The unit is shown.
pub fn {}(&self) -> {} {{
    match self {{
        {}
    }}
}}
"#, field_name, out_type, func_body)
    }

    let impl_body: String = COLUMNS.into_iter().map(|(&data, &(field, type_, unit))| {
        make_function(data, field, type_, unit, &ptable).to_owned()
    }).collect::<Vec<_>>().join("\n\t\t");

    let impl_body = impl_body.replace("Some(Time::new::<year>(\"Stable\".))", "None");

    let dest_path = Path::new(&out_dir).join("element_data.rs");
    std::fs::write(dest_path, format!(r#"
#[allow(clippy::excessive_precision)]
impl Element {{
    {}
}}
"#, impl_body)).unwrap();

    println!("cargo:rerun-if-changed=data/ptable.json");
    println!("cargo:rerun-if-changed=codegen/build.rs");

    Ok(())
}