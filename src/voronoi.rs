//! Implements the Voronoi cell construction used to find generators, relations,
//! an ASU for a space group.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    ops::{Add, Neg},
};

use nalgebra::{DMatrix, Matrix3, MatrixXx4, Point3, Vector3, Vector4};
use num_traits::Signed;

use crate::{
    frac, fract::Frac, geometry::Plane, isometry::Isometry, spacegroupdata::SpaceGroupSetting,
    symmop::SymmOp,
};

/// Shifts the point to lie within [0, 1) in all directions.
fn mod_one_pt(pt: &Point3<Frac>) -> Point3<Frac> {
    Point3::new(pt.x.modulo_one(), pt.y.modulo_one(), pt.z.modulo_one())
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum IntersectionResult {
    Inside,
    Outside,
    OnBoundary(Vec<usize>),
}

/// Finds a general position, ideally relatively close to the origin and with a
/// small denominator. Note that the algorithm needs to be adjusted when the
/// point lies on a symmetry element, hence why we avoid doing so.
pub fn find_suitable_origin(group: &SpaceGroupSetting) -> Point3<Frac> {
    let choices = vec![
        frac!(0),
        frac!(1 / 2),
        frac!(1 / 4),
        frac!(1 / 6),
        frac!(3 / 4),
        frac!(1 / 8),
    ];

    let op_isos = group.op_isometries();

    for i in 1..choices.len() {
        for x in 0..i {
            for y in 0..i {
                for z in 0..i {
                    let pt = Point3::new(choices[x], choices[y], choices[z]);

                    let pt_set: HashSet<Point3<Frac>> = op_isos
                        .iter()
                        .map(|j| mod_one_pt(&j.transform(&pt)))
                        .collect();

                    match pt_set.len().cmp(&op_isos.len()) {
                        Ordering::Less => continue,
                        Ordering::Equal => return pt,
                        Ordering::Greater => unreachable!("Can't have more points than isometries"),
                    }
                }
            }
        }
    }

    panic!("Failed to find a point")
}

/// Computes the Dirichlet-Voronoi cell points of the space group using the
/// given origin.
pub fn dv_cell(group: &SpaceGroupSetting, ori: Point3<Frac>) -> HashMap<Point3<Frac>, Vec<SymmOp>> {
    // behold, the worst possible algorithm for this
    let mut ops = group.all_symmops();
    let op_isos: Vec<Isometry> = ops.iter().map(|i| i.to_iso(group.is_hex())).collect();

    // The starting point for our algorithm is the translation ops. The
    // intersection of these planes defines a unit cube centered around our
    // point. Every operation has a unique shift inside this region, taking the
    // right boundary as open.
    let mut images: Vec<Option<Point3<Frac>>> = op_isos
        .iter()
        .map(|i| {
            let im = i.transform(&ori);
            if im == ori {
                // identity op
                None
            } else {
                // map each point within the bounds [-0.5, 0.5)
                Some(Point3::from(
                    (im.coords - ori.coords)
                        .add_scalar(frac!(1 / 2))
                        .map(|f| f.modulo_one())
                        .add_scalar(frac!(-1 / 2))
                        .add(ori.coords),
                ))
            }
        })
        .collect();

    let translations = &[
        Vector3::x(),
        Vector3::x().neg(),
        Vector3::y(),
        Vector3::y().neg(),
        Vector3::z(),
        Vector3::z().neg(),
    ]
    .map(SymmOp::Translation);

    // we want to avoid modulo behavior here: x + 1 and x - 1 are valid returns
    // here, it's just the linear generators that get mapped into mod_one space
    ops.extend_from_slice(translations);
    images.extend(
        translations
            .into_iter()
            .map(|tau| Some(tau.transform(&ori, group.is_hex()))),
    );

    let ref_planes: Vec<Option<Plane>> = images
        .iter()
        .map(|opt| opt.and_then(|im| Plane::from_opposite_points(&im, &ori)))
        .collect();

    let mut inds = vec![];
    let mut halfspaces = vec![];
    for (i, pl) in ref_planes.iter().enumerate() {
        if let Some(pl) = pl {
            let n = pl.normal().as_vec3();
            let d = pl.origin().dot(&n);
            // equation n ⋅ v - d = 0
            // if n ⋅ o < d, then <, otherwise >
            // flip all rows to be <
            halfspaces.extend(if n.dot(&ori.coords) > d {
                [-n.x, -n.y, -n.z, d]
            } else {
                [n.x, n.y, n.z, -d]
            });
            inds.push(i);
        };
    }

    let half_m = MatrixXx4::from_row_iterator(halfspaces.len() / 4, halfspaces).scale(frac!(6));

    println!("{}", half_m);

    fn test(
        half_m: &MatrixXx4<Frac>,
        ori: &Point3<Frac>,
        x: &Frac,
        y: &Frac,
        z: &Frac,
    ) -> IntersectionResult {
        let results = half_m * Vector4::new(ori.x + *x, ori.y + *y, ori.z + *z, frac!(1));
        let mut plane_inds = vec![];
        for (i, result) in results.iter().enumerate() {
            match result.cmp(&frac!(0)) {
                Ordering::Less => continue,
                Ordering::Equal => plane_inds.push(i),
                Ordering::Greater => return IntersectionResult::Outside,
            }
        }

        if plane_inds.len() <= 2 {
            IntersectionResult::Inside
        } else {
            IntersectionResult::OnBoundary(plane_inds)
        }
    }

    assert_eq!(
        test(&half_m, &ori, &frac!(0), &frac!(0), &frac!(0)),
        IntersectionResult::Inside
    );

    let mut fracs: Vec<Frac> = (-(Frac::DENOM / 2)..(Frac::DENOM / 2))
        .map(Frac::new_with_numerator)
        .collect();

    fracs.sort_by_key(|f| f.abs());

    let mut boundary_verts = HashMap::new();

    for x in &fracs {
        for y in &fracs {
            for z in &fracs {
                match test(&half_m, &ori, x, y, z) {
                    IntersectionResult::Inside => {}
                    IntersectionResult::Outside => {}
                    IntersectionResult::OnBoundary(inds) => {
                        boundary_verts.insert(Point3::new(x.clone(), y.clone(), z.clone()), inds);
                    }
                }
            }
        }
    }

    boundary_verts
        .into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|ind| ops[ind]).collect()))
        .collect()
}
