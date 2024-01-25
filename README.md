
# Curium

## 🦀💎 Crystal structure and symmetry for humans 💎🦀

**Disclaimer: this is still very much a work in progress. Large breaking changes may come without
notice!**

Curium (or, at least, what I hope Curium will be soon!) has two goals:

- 🔥 Blazing speed. Right now, a large amount of performance-critical calculation—file I/O, distance
  checks, generating space group operations, enumerating Wyckoff positions—is done with painfully
  slow amounts of Python code due to its dominance in the scientific landscape. It's my hope that a
  Python wrapper for this library will remove this bottleneck, just as LLM tokenizers have been
  rewritten in Rust to remove that bottleneck and other libraries like `numpy` use low-level code to
  get C speed with Python syntax.
- 🔬 An API that makes understanding crystals easier. Existing libraries are often written for
  specific workflows and intended for use by experienced materials scientists. My hope is that
  Curium can be accessible to people like me—AI researchers trying to understand this exciting
  domain without deep background in the subject. Many crystal libraries have weakly-typed APIs, poor
  documentation, and enormous libraries of extremely specific algorithms without any kind of entry
  point for a user. My goal is to make working with crystals natural for the user, even
  (especially!) if the user is a novice.

I think both of these goals are nicely summarized by my first work—implementing a description of
symmetry operations. Comparing with `SymmOp`, the `pymatgen` equivalent:

- 🔥 Curium can store symmetry operations compactly: due to more granular numeric types and
  sophisticated mathematical representations, a Curium `SymmOp` is about 10x smaller than the
  equivalent `pymatgen` `SymmOp`.
- 🔥 Curium avoids using floating-point arithmetic to represent symmetry operations. This means a
  single Curium `SymmOp` is actually more like 2 Python numbers, because space isn't wasted storing
  decimal places that aren't even correct in the first place.
- 🔬 Curium can represent generalized affine transformations (via the `Isometry` type), but for
  symmetry operations in groups Curium implements the algorithm given in ITA to turn those matrix
  operations into geometric operations. This means users of Curium can easily figure out what it
  represents beyond a list of numbers.
- 🔬 With `SymmOp`s defined entirely using integers, there's never any numerical stability issues,
  which can create challenging bugs. If you want to know if two operations are equal, just compare
  them directly!



## Roadmap

### Rust

Planned functionality:
- 🟨 Symmetry operations
- ❌ Composition
- ❌ Species
- ❌ Element
- ❌ Lattice
- ❌ CIF parsing
- ❌ Space groups
- ❌ Wyckoff positions
- ❌ PeriodicSite equivalent
- ❌ Structure

### Python
As the Rust side of things nears MVP status, a Python wrapper library is planned.

The Python library will have some Python interop functionality that's not planned for the Rust code
base:

- Pandas extensions so you can operate on `DataFrame`s containing Curium types easily
- Conversion to/from pymatgen, pyxtal, gemmi, ase, etc.
- Visualization tools!
## Contributing

✨ Contributions are always welcome! ✨

With the project in such an embryonic stage, it's probably best to contact me directly if you're
interested in building something within Curium. You can find me on here or, for more rapid
communication, find me on Discord at `pollardsrho`.


## Acknowledgements

 - This would not be possible without the *International Tables for Crystallography Volume A.* It's
   an exceptional reference, and my hope with Curium is to make crystallography code as clear as the
   book's exposition.

- I'm part of the [Machine Learning and Evolution Laboratory](https://mleg.cse.sc.edu/)
  `@usccolumbia` at the University of South Carolina. This code is, for the time being, purely my
  own—if it's bad, blame me.

- The authors of `pymatgen` have set out the roadmap for what Curium aims to do. Curium's API, at
  least in the broad strokes, is heavily indebted to their work in that library. GEMMI, PyXtal, and
  spglib are other libraries that I've looked at for inspiration.