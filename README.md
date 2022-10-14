MyQLM fermion module
=============================

`MyQLM-fermion` is the new open-source module in `MyQLM`.

This module contains tools specific to spin and fermionic systems. It includes, among other things:
- various objects to describe fermionic and spin Hamiltonians,
- objects to describe molecular systems,
- various transformations between fermionic and spin representations,
- a variational quantum eigensolver (VQE),
- a trotterization module,
- a quantum phase estimation module,
- a natural gradient-based optimizer,
- a sequential optimizer,
- a multiqubit gate noise mitigation plugin,
- an ADAPT-VQE plugin implementation,
- ...

Installation
----------------

For now, this module has to be installed directly from the sources:

```python
git clone https://github.com/myQLM/myqlm-fermion.git
cd myqlm-fermion
pip install -r requirements.txt
pip install .
```

Documentation
-------------

`MyQLM-fermion` is part of the MyQLM package, whose documentation can be found [here](https://myqlm.github.io/).
To access directly MyQLM-fermion documentation, [click here](https://myqlm.github.io/qat-fermion.html).

Changelog
---------

The changelog can be found [here](CHANGELOG.md).
