/// Testing functions: generates every space group.
use curium::{
    constants::{GROUP_ORDERS, HALL_SYMBOLS},
    hall::HallGroupSymbol,
    markup::ITA,
};

fn main() {
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
        coz::progress!();
    }
}
