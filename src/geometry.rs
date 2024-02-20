//! Geometry primitives.

use std::{cmp::Ordering, fmt::Display};

use nalgebra::{matrix, Matrix3, Matrix4, Point3, Vector3};
use num_traits::Zero;
use simba::scalar::SupersetOf;

use crate::{
    frac,
    fract::{BaseInt, Frac},
    markup::{Block, RenderBlocks},
};

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
        if v.is_zero() {
            return Some(frac!(0));
        }
        let tau = v;
        let full_tau = self.as_vec3();
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

    /// Gets a rotation axis through which 180-degree rotation followed by inversion would produce
    /// the same result as reflection through this plane.
    pub fn normal_axis(&self) -> RotationAxis {
        RotationAxis::new_with_dir(self.n, Point3::origin() + self.origin())
    }
}

/// A symmetry element: an affine subspace of R3. This can be either a single point, a line, a
/// plane, or all of R3.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymmetryElement {
    /// A single point: the inversion center.
    Point(Point3<Frac>),
    /// A line: the rotation axis.
    Line(RotationAxis),
    /// A plane: the reflection plane.
    Plane(Plane),
    /// All of R3, for the identity.
    Space,
}

impl SymmetryElement {
    /// Gets an affine matrix that maps (x, y, z, 1) to the symmetry element. The axes are mapped so
    /// that, where not restricted by dependencies, axes map directly to their image. For example,
    /// the yz-plane is represented as 0, y, z, but the axis 111 is represented by x, x, x.
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
}

impl RenderBlocks for SymmetryElement {
    fn components(&self) -> Vec<Block> {
        if self == &Self::Space {
            vec![]
        } else {
            self.to_iso().components()
        }
    }
}
