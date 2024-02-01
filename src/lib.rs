// frac!(-1) gets expanded to frac!(-1 * Frac::DENOM). I don't want to change this.
#![allow(clippy::neg_multiply)]

#[macro_use]
extern crate uom;

pub mod element;
pub mod frac;
pub mod isometry;
pub mod lattice;
pub mod markup;
pub mod symbols;
pub mod symmop;
pub mod units;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
