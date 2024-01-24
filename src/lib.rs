#[macro_use]
extern crate uom;

pub mod frac;
pub mod isometry;
pub mod symmop;
pub mod units;
pub mod element;

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
