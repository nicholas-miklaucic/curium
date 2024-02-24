//! Geometry primitives.

use std::{cmp::Ordering, fmt::Display, ops::BitAnd};

use nalgebra::{matrix, Matrix3, Matrix4, Point3, Vector3};
use num_traits::Zero;
use simba::scalar::SupersetOf;

use crate::{
    frac,
    fract::{BaseInt, Frac},
    markup::{Block, RenderBlocks, ITA},
    symbols::{EMPTY_SET, RR, SUP_3},
};

/// Returns the number d such that d * v1 = v2, if one exists.
fn compute_vec_ratio(v1: Vector3<Frac>, v2: Vector3<Frac>) -> Option<Frac> {
    if v2.is_zero() {
        return Some(frac!(0));
    }

    if v1.is_zero() {
        return None;
    }
    let tau = v2;
    let full_tau = v1;
    let mut scale = None;
    for i in 0..3 {
        let full_i = full_tau[i];
        let tau_i = tau[i];
        if tau_i == full_i && full_i.is_zero() {
            // this axis could be any scale
            continue;
        }
        // dbg!(self, v, tau, full_tau);
        let new_scale = tau_i / full_i;
        if scale.is_some_and(|f| f != new_scale) {
            // this could be an error in the future, perhaps
            // mismatching is not good!
            return None;
        }
        scale.replace(new_scale);
    }

    scale
}

/// A direction, such as in a rotation axis or normal vector.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct Direction {
    /// The first lattice point from the origin in this direction.
    v: Vector3<i8>,
}

impl Direction {
    /// Creates a new Direction from a general vector.
    pub fn new(vec: Vector3<Frac>) -> Self {
        /// Computes the GCD between a and b, treating 0 as null.
        fn recip_gcd(a: BaseInt, b: BaseInt) -> BaseInt {
            match (a.abs(), b.abs()) {
                (0, b_abs) => b_abs,
                (a_abs, 0) => a_abs,
                (a_abs, b_abs) => Frac::gcd(a_abs, b_abs),
            }
        }
        let nums: Vec<BaseInt> = vec.iter().map(|&f| f.numerator).collect();
        let scale = nums.clone().into_iter().reduce(recip_gcd).unwrap();

        Self {
            v: Vector3::new(
                (nums[0] / scale) as i8,
                (nums[1] / scale) as i8,
                (nums[2] / scale) as i8,
            ),
        }
    }

    /// The `Direction` represented as a vector of `Frac`s.
    pub fn as_vec3(&self) -> Vector3<Frac> {
        Vector3::new(
            frac!(self.v.x as i16),
            frac!(self.v.y as i16),
            frac!(self.v.z as i16),
        )
    }

    /// Returns whether the axis is correctly oriented for ITA conventions.
    pub fn is_conventionally_oriented(&self) -> bool {
        // ????? this rule is not explained anywhere I can find. After some reverse-engineering, it
        // appears to be that an even number of negative signs are preferred, unless it's like [0,
        // -1, -1], in which case flipping is clearly better.

        let mut num_neg = 0;
        let mut num_pos = 0;
        let mut first_is_pos: Option<bool> = None;
        for i in 0..3 {
            match self.v[i].cmp(&0) {
                Ordering::Less => {
                    first_is_pos = first_is_pos.or(Some(false));
                    num_neg += 1;
                }
                Ordering::Equal => {}
                Ordering::Greater => {
                    first_is_pos = first_is_pos.or(Some(true));
                    num_pos += 1;
                }
            }
        }

        if num_pos == num_neg {
            first_is_pos.unwrap()
        } else {
            (num_neg % 2 == 0) && (num_pos > 0)
        }
    }

    /// Flips the direction.
    pub fn inv(&self) -> Self {
        Self {
            v: Vector3::new(-self.v.x, -self.v.y, -self.v.z),
        }
    }

    /// Gets a scaled version of the direction's first lattice vector.
    pub fn scaled_vec(&self, scale: Frac) -> Vector3<Frac> {
        self.as_vec3().scale(scale)
    }

    pub fn compute_scale(&self, v: Vector3<Frac>) -> Option<Frac> {
        compute_vec_ratio(self.as_vec3(), v)
    }

    /// Converts to cartesian Miller indices from the hexagonal version.
    pub fn to_cart(&self) -> Direction {
        let m: Matrix3<Frac> = matrix![
            frac!(2), frac!(1), frac!(0);
            frac!(1), frac!(2), frac!(0);
            frac!(0), frac!(0), frac!(1)
        ];
        Direction::new(m * self.v.map(Frac::from))
    }

    /// Converts to hexagonal Miller indices.
    pub fn to_hex(&self) -> Direction {
        let m: Matrix3<Frac> = matrix![
            frac!(-2/3), frac!(1/3), frac!(0);
            frac!(-1/3), frac!(2/3), frac!(0);
            frac!(0), frac!(0), frac!(1)
        ];
        Direction::new(m * self.v.map(Frac::from))
    }

    /// If needed, orients to conform to ITA conventions. Use only when forward and backward are
    /// equivalent.
    pub fn conventional_orientation(&self) -> Self {
        if self.is_conventionally_oriented() {
            *self
        } else {
            self.inv()
        }
    }

    /// Gets the standard basis vectors for the plane normal to this direction. This is defined as
    /// the smallest integral vectors b1, b2 such that b1 x b2 is parallel to the normal vector.
    pub fn plane_basis(&self) -> (Direction, Direction) {
        let mut num_zeros = 0;
        let mut num_nonzeros = 0;
        let mut b1 = vec![frac!(0), frac!(0), frac!(0)];
        let mut b2 = vec![frac!(0), frac!(0), frac!(0)];
        let mut b3 = vec![frac!(0), frac!(0), frac!(0)];
        for i in 0..3 {
            let el = self.as_vec3()[i];
            if el.is_zero() {
                if num_zeros == 0 {
                    b1[i] = frac!(1);
                } else {
                    b2[i] = frac!(1);
                }
                num_zeros += 1;
            } else {
                b3[i] = match num_nonzeros {
                    0 => frac!(1) / el,
                    1 => frac!(-1) / el,
                    _ => frac!(0),
                };
                num_nonzeros += 1;
            }
        }

        let (e1, e2) = match num_zeros {
            0 => {
                // challenging case e.g., [a b c] nonzero
                // b3 is [1/a -1/b 0], which is normal
                // cross product is [c/b c/a -a/b - b/a]
                // scaling by ab, we get [ac bc -(a^2 + b^2)]
                // which is integral
                let [a, b, c] = *self.v.as_slice() else {
                    panic!("v is 3D");
                };
                (
                    Direction::new(Vector3::from_row_slice(&b3)),
                    Direction::new(Vector3::new(
                        frac!(a * c),
                        frac!(b * c),
                        Frac::from(-(a * a + b * b)),
                    )),
                )
            }
            1 => (
                Direction::new(Vector3::from_row_slice(&b1)),
                Direction::new(Vector3::from_row_slice(&b3)),
            ),
            2 => (
                Direction::new(Vector3::from_row_slice(&b1)),
                Direction::new(Vector3::from_row_slice(&b2)),
            ),
            _ => panic!("Zero direction is not allowed"),
        };

        (e1.conventional_orientation(), e2.conventional_orientation())
    }

    /// Returns whether the two directions are parallel.
    pub fn is_parallel(&self, other: Direction) -> bool {
        self.v == other.v || self.v == -other.v
    }
}

impl Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nums = format!("{}{}{}", self.v.x, self.v.y, self.v.z);
        let nums = nums.replace('-', "\u{0305}");
        write!(f, "[{nums}]")
    }
}

/// The axis of rotation. Distinguishes a single point on the line, which is the center of a
/// potential rotoinversion.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct RotationAxis {
    /// The origin.
    origin: Point3<Frac>,
    /// The direction of the axis.
    dir: Direction,
}

impl RotationAxis {
    /// Vector oriented as v going through origin.
    pub fn new(v: Vector3<Frac>, origin: Point3<Frac>) -> Self {
        Self {
            origin,
            dir: Direction::new(v),
        }
    }

    /// Direction of axis.
    pub fn dir(&self) -> Direction {
        self.dir
    }

    /// Direction of axis, as a vector.
    pub fn dir_vec(&self) -> Vector3<Frac> {
        self.dir.as_vec3()
    }

    /// Origin of axis.
    pub fn origin(&self) -> Point3<Frac> {
        self.origin
    }

    /// Direction and origin.
    pub fn new_with_dir(dir: Direction, origin: Point3<Frac>) -> Self {
        Self { origin, dir }
    }

    /// Return representation as origin and vector.
    pub fn as_origin_vector(&self) -> (Point3<Frac>, Vector3<Frac>) {
        (self.origin, self.dir.as_vec3())
    }

    /// Returns whether the axis is correctly oriented for ITA conventions.
    pub fn is_conventionally_oriented(&self) -> bool {
        self.dir.is_conventionally_oriented()
    }

    /// Returns a "conventional" orientation for the axis. Note that, if you're using this to
    /// represent a generic ray, this can flip the direction of the ray.
    pub fn conventional(&self) -> Self {
        if self.is_conventionally_oriented() {
            *self
        } else {
            self.inv()
        }
    }

    /// Returns an axis going in the other direction. The origin remains unchanged.
    pub fn inv(&self) -> Self {
        Self {
            origin: self.origin,
            dir: self.dir.inv(),
        }
    }

    /// Moves the axis modulo a unit cell.
    pub fn modulo_unit_cell(&self) -> Self {
        Self {
            origin: self.origin.map(|f| f.modulo_one()),
            dir: self.dir,
        }
    }

    /// Returns whether a point lies on the axis.
    pub fn contains(&self, point: Point3<Frac>) -> bool {
        point == self.origin
            || (Direction::new(point.coords - self.origin.coords).conventional_orientation()
                == self.dir.conventional_orientation())
    }
}

/// A plane (particularly of reflection.)
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct Plane {
    /// The direction of the normal vector.
    n: Direction,
    /// The scale such that nd is on the plane.
    d: Frac,
    /// The origin.
    ori: Point3<Frac>,
}

impl Plane {
    /// Initializes a plane from its basis and a point on it. Specifically, this plane is the set {o
    /// + a v1 + b v2}, for all real a, b.
    pub fn from_basis_and_origin(
        v1: Vector3<Frac>,
        v2: Vector3<Frac>,
        origin: Point3<Frac>,
    ) -> Self {
        let normal = v1.cross(&v2);
        let n = Direction::new(normal);
        // println!("{} {}", n, origin);
        let dist = n.as_vec3().dot(&origin.coords);
        Self {
            n,
            d: dist,
            ori: origin,
        }
    }

    /// Constructs a plane from a point through it and a normal vector.
    pub fn from_point_and_normal(pt: Point3<Frac>, normal: Vector3<Frac>) -> Self {
        Self {
            n: Direction::new(normal),
            d: normal.dot(&pt.coords),
            ori: pt,
        }
    }

    /// Constructs the plane of points equidistant from the two inputs: the perpendicular bisector
    /// in 3D. Returns `None` if the two points are identical.
    pub fn from_opposite_points(p1: &Point3<Frac>, p2: &Point3<Frac>) -> Option<Self> {
        if p1 == p2 {
            None
        } else {
            let normal = (p1.coords - p2.coords) * frac!(24);
            let mid = (p1.coords + p2.coords).scale(frac!(1 / 2));
            Some(Self::from_point_and_normal(mid.into(), normal))
        }
    }

    pub fn reflection_matrix(&self) -> Matrix4<f64> {
        // for now, use f64

        // https://www.wikiwand.com/en/Transformation_matrix#Reflection_2
        let fl_n: Vector3<f64> = self.n.as_vec3().to_subset_unchecked();
        let fl_n_norm = fl_n.norm();
        let &[a, b, c] = fl_n.scale(1. / fl_n_norm).as_slice() else {
            panic!()
        };
        let d: f64 = self.d.to_subset_unchecked();
        let d = d / fl_n_norm;
        let abcd = [a, b, c, d];
        Matrix4::from_fn(|i, j| {
            let entry = if i == j { 1. } else { 0. };
            if i == 3 {
                entry
            } else {
                entry - 2. * abcd[i] * abcd[j]
            }
        })
    }

    /// Returns an equivalent representation but oriented the opposite direction.
    pub fn inv(&self) -> Self {
        Self {
            n: self.n.inv(),
            d: -self.d,
            ori: self.ori,
        }
    }

    /// Shifts the plane by unit cells so it lies as close to the origin as possible.
    pub fn modulo_unit_cell(&self) -> Plane {
        Plane {
            n: self.n,
            d: std::cmp::max(self.d.modulo_one(), -(self.d.modulo_one())),
            ori: self.ori.map(|f| f.modulo_one()),
        }
    }

    /// Gets the basis vectors of the plane. The two should be orthogonal.
    pub fn basis_vectors(&self) -> (Vector3<Frac>, Vector3<Frac>) {
        let (b1, b2) = self.n.plane_basis();
        (b1.as_vec3(), b2.as_vec3())
    }

    /// Gets a point on the plane.
    pub fn origin(&self) -> Vector3<Frac> {
        self.ori.coords
    }

    /// Gets the conventional orientation of the normal vector. Does not change the plane's
    /// geometric meaning.
    pub fn conventional(&self) -> Self {
        if self.n.is_conventionally_oriented() {
            *self
        } else {
            self.inv()
        }
    }

    /// Gets the normal direction of the plane.
    pub fn normal(&self) -> Direction {
        self.n
    }

    /// Gets a normal vector of the plane.
    pub fn normal_vec(&self) -> Vector3<Frac> {
        self.n.as_vec3()
    }

    /// Gets a rotation axis through which 180-degree rotation followed by inversion would produce
    /// the same result as reflection through this plane.
    pub fn normal_axis(&self) -> RotationAxis {
        RotationAxis::new_with_dir(self.n, Point3::origin() + self.origin())
    }

    /// Whether the plane contains a point.
    pub fn contains_pt(&self, pt: Point3<Frac>) -> bool {
        pt.coords.dot(&self.normal_vec()) == self.d()
    }

    /// The d such that the plane's equation is n ⋅ v - d = 0.
    pub fn d(&self) -> Frac {
        self.origin().dot(&self.normal_vec())
    }
}

/// A symmetry element: an affine subspace of R3. This can be either the empty set, a single point,
/// a line, a plane, or all of R3.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SymmetryElement {
    /// The empty set.
    Null,
    /// A single point.
    Point(Point3<Frac>),
    /// A line.
    Line(RotationAxis),
    /// A plane.
    Plane(Plane),
    /// All of R3.
    Space,
}

impl SymmetryElement {
    /// Gets an affine matrix that maps (x, y, z, 1) to the symmetry element. The axes are mapped so
    /// that, where not restricted by dependencies, axes map directly to their image. For example,
    /// the yz-plane is represented as 0, y, z, but the axis 111 is represented by x, x, x. Returns
    /// the all-zero matrix for `Null`.
    pub fn to_iso(&self) -> Matrix4<Frac> {
        let f0 = frac!(0);
        let f1 = frac!(1);
        let mut aff = matrix![
            f0, f0, f0, f0;
            f0, f0, f0, f0;
            f0, f0, f0, f0;
            f0, f0, f0, f1;
        ];
        match *self {
            SymmetryElement::Null => {
                aff.fill_diagonal(f0);
            }
            SymmetryElement::Point(o) => {
                aff[(0, 3)] = o.x;
                aff[(1, 3)] = o.y;
                aff[(2, 3)] = o.z;
            }
            SymmetryElement::Line(axis) => {
                let dir = axis.dir().as_vec3();
                // get letter to use for axis: first nonzero direction
                let mut base_ax = 0;
                while dir[base_ax].is_zero() {
                    base_ax += 1;
                }

                aff.fixed_view_mut::<3, 1>(0, base_ax)
                    .copy_from_slice(dir.as_slice());

                let o = axis.origin();
                aff.fixed_view_mut::<3, 1>(0, 3)
                    .copy_from_slice(&[o.x, o.y, o.z]);

                // dbg!(axis, dir);
            }
            SymmetryElement::Plane(plane) => {
                let (v1, v2) = plane.basis_vectors();
                let (a1, _f1) = v1.iter().enumerate().find(|(_i, e)| !e.is_zero()).unwrap();

                let (a2, _f2) = v2
                    .iter()
                    .enumerate()
                    .find(|(i, e)| i != &a1 && !e.is_zero())
                    .unwrap();

                aff.fixed_view_mut::<3, 1>(0, a1)
                    .copy_from_slice(v1.as_slice());
                aff.fixed_view_mut::<3, 1>(0, a2)
                    .copy_from_slice(v2.as_slice());
                aff.fixed_view_mut::<3, 1>(0, 3)
                    .copy_from_slice(plane.origin().as_slice())
            }
            SymmetryElement::Space => {
                aff.fill_diagonal(f1);
            }
        }

        aff
    }

    pub fn contains_pt(&self, pt: Point3<Frac>) -> bool {
        match *self {
            SymmetryElement::Null => false,
            SymmetryElement::Point(p) => p == pt,
            SymmetryElement::Line(l) => l.contains(pt),
            SymmetryElement::Plane(pl) => pl.contains_pt(pt),
            SymmetryElement::Space => true,
        }
    }

    pub fn new_point(pt: Point3<Frac>) -> Self {
        SymmetryElement::Point(pt)
    }

    pub fn new_line(origin: Point3<Frac>, dir_vec: Vector3<Frac>) -> Self {
        SymmetryElement::Line(RotationAxis::new(dir_vec, origin))
    }

    pub fn new_plane(normal: Vector3<Frac>, origin: Point3<Frac>) -> Self {
        SymmetryElement::Plane(Plane::from_normal_and_origin(normal, origin))
    }

    pub fn intersection(self, other: Self) -> Self {
        self & other
    }
}

impl BitAnd for SymmetryElement {
    type Output = SymmetryElement;

    /// Computes the intersection of two symmetry elements.
    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (SymmetryElement::Null, _) | (_, SymmetryElement::Null) => SymmetryElement::Null,
            (SymmetryElement::Space, other) | (other, SymmetryElement::Space) => other,
            (pt @ SymmetryElement::Point(p), other) | (other, pt @ SymmetryElement::Point(p)) => {
                if other.contains_pt(p) {
                    pt
                } else {
                    SymmetryElement::Null
                }
            }
            (el1 @ SymmetryElement::Line(l1), _el2 @ SymmetryElement::Line(l2)) => {
                let (o1, v1) = (l1.origin().coords, l1.dir_vec());
                let (o2, v2) = (l2.origin().coords, l2.dir_vec());
                // o1 + a v1 = o2 + b v2
                // a v1 = (o2 - o1) + b v2
                // a (v1 x v2) = (o2 - o1) x v2   <-- v2 x v2 = 0, so this cancels
                let v1xv2 = v1.cross(&v2);
                let doxv2 = (o2 - o1).cross(&v2);

                let parallel = v1xv2.is_zero();
                let incident = doxv2.is_zero();
                if parallel && incident {
                    // lines are equivalent
                    el1
                } else if parallel {
                    // lines are parallel but never coincide
                    SymmetryElement::Null
                } else {
                    // ((o2 - o1) x v2) / (v1 x v2) must be equal for all elements

                    // println!("{o1} + a {v1} = {o2} + b {v2}");

                    match compute_vec_ratio(v1xv2, doxv2) {
                        Some(scale) => {
                            // dbg!(scale, v1xv2, doxv2);
                            SymmetryElement::new_point(Point3::origin() + o1 + v1.scale(scale))
                        }
                        None => SymmetryElement::Null,
                    }
                }
            }

            (el1 @ SymmetryElement::Plane(pl1), el2 @ SymmetryElement::Plane(pl2)) => {
                let parallel = pl1.normal().is_parallel(pl2.normal());
                let incident = pl1.contains_pt(Point3::origin() + pl2.origin());
                if parallel && incident {
                    // same plane
                    el1
                } else if parallel {
                    // never meet, parallel but apart
                    SymmetryElement::Null
                } else {
                    // non-parallel planes always meet at a line with direction n1 x n2

                    // need to find an intersection point: we can find the point on the line made by
                    // o1 and a basis vector that isn't parallel to the plane.
                    let (b1, b2) = pl1.basis_vectors();
                    let basis_vec = if b1.dot(&pl2.normal_vec()).is_zero() {
                        b2
                    } else {
                        b1
                    };

                    let basis_line = SymmetryElement::new_line(Point3::origin(), basis_vec);

                    if let SymmetryElement::Point(p) = basis_line & el2 {
                        SymmetryElement::new_line(p, pl1.normal_vec().cross(&pl2.normal_vec()))
                    } else {
                        panic!("Line should have intersected: {:?} {:?}", el1, el2);
                    }
                }
            }

            (el1 @ SymmetryElement::Line(l), SymmetryElement::Plane(pl))
            | (SymmetryElement::Plane(pl), el1 @ SymmetryElement::Line(l)) => {
                let parallel = l.dir_vec().dot(&pl.normal_vec()).is_zero();
                let incident = pl.contains_pt(l.origin());
                if parallel && incident {
                    // coincide
                    el1
                } else if parallel {
                    // never intersect
                    SymmetryElement::Null
                } else {
                    //
                    // n * (o + k v) = d
                    // n * o + k(n * v) = d
                    // k = (d - n * o) / (n * v)
                    let o = l.origin();
                    let v = l.dir_vec();
                    let n = pl.normal_vec();
                    let d = pl.d();

                    let k = (d - n.dot(&o.coords)) / n.dot(&v);
                    let line_origin = o + v.scale(k);

                    SymmetryElement::Point(line_origin)
                }
            }
        }
    }
}

impl RenderBlocks for SymmetryElement {
    fn components(&self) -> Vec<Block> {
        match *self {
            Self::Space => vec![RR, SUP_3],
            Self::Null => vec![EMPTY_SET],
            s => s.to_iso().components(),
        }
    }
}

impl Display for SymmetryElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", ITA.render_to_string(self))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::vector;

    use super::*;

    #[test]
    fn test_plane_plane_intersection() {
        let s1 =
            SymmetryElement::new_plane(vector![frac!(1), frac!(-1), frac!(0)], Point3::origin());

        let s2 =
            SymmetryElement::new_plane(vector![frac!(1), frac!(1), frac!(0)], Point3::origin());

        assert_eq!(
            s1 & s2,
            SymmetryElement::new_line(Point3::origin(), Vector3::z())
        );
    }

    #[test]
    fn test_line_line_intersection() {
        let s1 =
            SymmetryElement::new_line(Point3::origin(), vector![frac!(1), frac!(-1), frac!(0)]);

        let s2 = SymmetryElement::new_line(Point3::origin(), vector![frac!(1), frac!(1), frac!(0)]);

        assert_eq!(s1 & s2, SymmetryElement::new_point(Point3::origin()));

        let pt = Point3::new(frac!(1), frac!(1), frac!(0));
        let s3 = SymmetryElement::new_line(pt, vector![frac!(0), frac!(0), frac!(1)]);

        assert_eq!(s2 & s3, SymmetryElement::new_point(pt));
        assert_eq!(s1 & s3, SymmetryElement::Null);
    }

    #[test]
    fn test_line_plane_intersection() {
        let s1 = SymmetryElement::new_line(Point3::new(frac!(0), frac!(1), frac!(1)), Vector3::y());
        let s2 = SymmetryElement::new_plane(
            vector![frac!(1), frac!(-1), frac!(0)],
            Point3::new(frac!(1), frac!(1 / 2), frac!(1)),
        );
        assert_eq!(
            s1 & s2,
            SymmetryElement::new_point(Point3::new(frac!(0), frac!(-1 / 2), frac!(1)))
        );
        let s3 = SymmetryElement::new_plane(
            vector![frac!(-1), frac!(0), frac!(1)],
            Point3::new(frac!(-1), frac!(0), frac!(0)),
        );

        assert_eq!(s1 & s3, s1);
    }
}
