//! Defines base traits for abstract algebra concepts. Helps group (ha) related functionality together, but hopefully also sheds some light on how abstract algebra applies to Curium.

use std::{fmt::Debug, ops::Range};

/// Group element requirements.
pub trait GroupElement: Debug + Clone {}

impl<T: Debug + Clone> GroupElement for T {}

/// A mathematical group: a set and operation that satisfies closure, the existence of an identity,
/// the existence of an inverse, and associativity. Multiple groups, with different semantics, can
/// be defined for a single element type, and the group can operate under a different equivalence
/// relation than the default for the element type.
pub trait Group<E: GroupElement> {
    /// The identity. Must be an e such that ae = ea = a for all a in the group.
    fn identity(&self) -> E;

    /// Computing the inverse: must have ab = ba = e for a to be b's inverse.
    fn inv(&self, element: &E) -> E;

    /// The group operation. Must be associative. `g.compose(a, b)` returns `ab`, which is the
    /// operation "do b, then do a".
    fn compose(&self, a: &E, b: &E) -> E;

    /// Equivalence relation on group elements. For example, the group of integers modulo 7 doesn't
    /// use the standard [`std::cmp::Eq`] implementation and needs a custom implementation.
    fn equiv(&self, a: &E, b: &E) -> bool;

    /// "Canonical" or "reduced" representation of an element. If the group element has a natural
    /// equivalence relation given by [`Eq`], then we should have that, if `g.residue(a) ==
    /// g.residue(b)`, then `g.equiv(a, b)`. The default implementation (just a clone) is never
    /// wrong, but sometimes there's a natural choice here. For example, the integers mod 7 under
    /// addition can be represented by the numbers 0, 1, 2, 3, 4, 5, 6 by reducing modulo 7 in the
    /// standard sense.
    fn residue(&self, el: &E) -> E {
        el.clone()
    }

    /// Combines the elements using the composition operation, computing `abc` when given `[a, b,
    /// c]`. Returns the identity when given an empty list.
    fn reduce<'a, T: IntoIterator<Item = &'a E>>(&self, elems: T) -> E
    where
        E: 'a,
    {
        elems
            .into_iter()
            .fold(self.identity(), |acc, el| self.compose(&acc, el))
    }

    /// Conjugation by `a` of `b`, e.g., `aba^-1`.
    fn conjugate(&self, a: &E, b: &E) -> E {
        self.reduce([a, b, &self.inv(a)])
    }

    /// Containment using the group's notion of equality.
    fn contains_equiv<'a, T: IntoIterator<Item = &'a E>>(
        &self,
        elements: T,
        test_element: &'a E,
    ) -> bool {
        elements.into_iter().any(|el| self.equiv(&el, test_element))
    }
}

/// A group that is finitely generated.
pub trait FinitelyGeneratedGroup<E: GroupElement>: Group<E> {
    type Generators: IntoIterator<Item = E>;
    /// Gets the generators of the group. These elements should be able to produce every element in
    /// the group through composition and inversion.
    fn generators(&self) -> Self::Generators;
}

/// A finite group. Note that a finitely generated group can be infinite (e.g., the integers under
/// addition), but a finite group has finitely many generators.
pub trait FiniteGroup<E: GroupElement>: FinitelyGeneratedGroup<E> {
    type Elements: IntoIterator<Item = E>;
    /// Generates every element of the group. [`generate_elements`] implements this using just the
    /// generators, and the default implementation simply wraps that. Implementers should override
    /// this in one of three cases:
    /// - The implementer is a singleton and the outputs can be cached to avoid redoing the
    ///   (expensive) generation.
    /// - The maximum order of a generator is large enough that measures need to be taken to avoid
    ///   overflow, perhaps by avoiding internal collections and working functionally.
    /// - The output elements should adhere to some ordering principle that the default does not
    ///   respect, or there is a simpler potential implementation.
    fn elements(&self) -> Self::Elements;
}

/// An automated implementation of [`FiniteGroup::elements`] that loops through every generator
/// until a cycle is found and then combines all potential combinations of elements until closure is
/// achieved.
pub fn generate_elements<E: GroupElement, G: FinitelyGeneratedGroup<E>>(group: &G) -> Vec<E> {
    let gens: Vec<E> = group
        .generators()
        .into_iter()
        .flat_map(|e| [group.inv(&e), e])
        .collect();
    let mut elements = vec![group.identity()];
    let mut closure_achieved = false;

    while !closure_achieved {
        let el = elements.last().unwrap().clone();
        closure_achieved = true;
        for gen in &gens {
            // get all new elements that can be made by composing the generator
            // dbg!(&gen, &el);
            let mut combo = group.compose(&gen, &el);
            while !group.contains_equiv(&elements, &combo) {
                elements.push(combo.clone());
                combo = group.compose(&gen, &combo);
                closure_achieved = false;
            }
        }
    }

    elements
}

/// An automated implementation of [`FiniteGroup::elements`] that uses Dimino's algorithm.
pub fn generate_elements_dimino<E: GroupElement, G: FinitelyGeneratedGroup<E>>(
    group: &G,
) -> Vec<E> {
    let gens: Vec<E> = group
        .generators()
        .into_iter()
        .flat_map(|e| [group.residue(&e)])
        .collect();
    let mut elements = vec![group.identity()];

    for i in 0..gens.len() {
        // subgroup of G, G_i, given by gens[0..i]
        let d = elements.clone();
        let mut n = vec![group.identity()];

        while !n.is_empty() {
            let mut new_n = vec![];
            for a in n {
                for g in gens.iter().skip(i) {
                    let ag = group.compose(&a, g);
                    if !group.contains_equiv(&elements, &ag) {
                        // G_i * g
                        for el in &d {
                            let ap = group.compose(el, &ag);
                            elements.push(ap.clone());
                            new_n.push(ap);
                        }
                    }
                }
            }
            n = new_n;
        }
    }

    elements
}

/// The integers mod n over addition, represented using the integers 0-(n-1). Mainly used for
/// testing.
#[derive(Debug, Clone, Copy)]
struct ZAddMod(usize);

impl Group<usize> for ZAddMod {
    fn identity(&self) -> usize {
        0
    }

    fn inv(&self, element: &usize) -> usize {
        self.0 - element
    }

    fn compose(&self, a: &usize, b: &usize) -> usize {
        (a + b) % self.0
    }

    fn equiv(&self, a: &usize, b: &usize) -> bool {
        a % self.0 == b % self.0
    }

    fn residue(&self, el: &usize) -> usize {
        el % self.0
    }
}

impl FinitelyGeneratedGroup<usize> for ZAddMod {
    type Generators = Range<usize>;

    fn generators(&self) -> Self::Generators {
        1..2
    }
}

impl FiniteGroup<usize> for ZAddMod {
    type Elements = Vec<usize>;

    fn elements(&self) -> Self::Elements {
        generate_elements(self)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::{hall::HallGroupSymbol, markup::ITA, symmop::SymmOp};

    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_zadd_n() {
        for n in [7, 10, 256] {
            let grp = ZAddMod(n);
            let mut els = grp.elements();
            els.sort();
            assert_eq!(els, (0..n).into_iter().collect::<Vec<usize>>());
            assert_eq!(
                els,
                (n..n + n)
                    .into_iter()
                    .map(|i| grp.residue(&i))
                    .collect::<Vec<usize>>()
            );
        }
    }

    #[test]
    fn test_dimino() {
        let grp = HallGroupSymbol::from_str("P 4n 2 3 -1n").unwrap();
        let sg = grp.generate_group();

        let s1 = generate_elements(&sg);
        let s2 = generate_elements_dimino(&sg);

        let ops1: Vec<String> = s1
            .iter()
            .map(|x| ITA.render_to_string(&x.to_iso(false)))
            .collect();
        let ops2: Vec<String> = s2
            .iter()
            .map(|x| ITA.render_to_string(&x.to_iso(false)))
            .collect();

        println!("{}\n---\n{}", ops1.join("\n "), ops2.join("\n "));
        for s in &s1 {
            assert_eq!(&SymmOp::classify_affine(s.to_iso(false)).unwrap(), s);
            assert!(
                sg.contains_equiv(&s2, s),
                "{}",
                ITA.render_to_string(&s.to_iso(false))
            );
        }
        for s in &s2 {
            assert_eq!(&SymmOp::classify_affine(s.to_iso(false)).unwrap(), s);
            assert!(
                sg.contains_equiv(&s1, s),
                " {}",
                ITA.render_to_string(&s.to_iso(false))
            );
        }

        assert_eq!(s1.len(), s2.len());
    }
}
