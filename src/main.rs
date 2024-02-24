/// Testing functions: generates every space group.
use curium::{
    constants::{GROUP_ORDERS, HALL_SYMBOLS},
    hall::HallGroupSymbol,
    markup::{Block, DISPLAY, ITA},
    spacegroupdata::SpaceGroupSetting,
    voronoi::{dv_cell, find_suitable_origin},
};

#[allow(dead_code)]
fn run_groups() {
    // coz::scope!("main");
    for (hall, grp_num) in HALL_SYMBOLS {
        let group: HallGroupSymbol = hall.parse().unwrap();
        let setting = group.generate_group();
        // dbg!(&setting);
        // println!(
        //     "{:?}\n{}",
        //     setting.lattice_type,
        //     ITA.render_to_string(&setting.op_list())
        // );
        assert_eq!(
            setting.all_symmops().len(),
            GROUP_ORDERS[grp_num],
            "\nGroup {}\nHall: {}\n{:#?}\n{}\n{:#?} {:?}",
            grp_num,
            hall,
            group,
            ITA.render_to_string(&setting.op_list()),
            setting.lattice_type,
            setting.centering
        );
    }
}

fn main() {
    let setting = SpaceGroupSetting::from_number(220);
    println!("{}\n", DISPLAY.render_to_string(&setting.op_list()));

    let origin = find_suitable_origin(&setting);
    println!("Origin: {}", DISPLAY.render_to_string(&origin));

    let verts = dv_cell(&setting, origin);
    for (pt, _ops) in &verts {
        println!("{:>15}", DISPLAY.render_to_string(pt));
    }
}
