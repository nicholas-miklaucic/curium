//! Defines base traits for abstract algebra concepts. As of now, this doesn't have much planned as
//! far as generic functionality, because groups/rings are very different, but the hope is that
//! organizing related functionality with joint guarantees ensures consistency and highlights
//! correspondences.

/// A mathematical group: a set and operation that satisfies closure, the existence of an identity,
/// the existence of an inverse, and associativity.
pub trait Group {
    /// The identity. Must be an e such that ae = ea = a for all a in the group.
    fn identity() -> Self;

    /// Computing the inverse: must have fg = gf = e for f to be g's inverse.
    fn inv(&self) -> Self;

    /// The group operation. Must be associative. `f.op(g)` returns `fg`.
    fn op(&self, rhs: &Self) -> Self;
}
