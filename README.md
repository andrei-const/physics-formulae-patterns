# physics-formulae-patterns
Statistical Patterns in the Equations of Physics

[![arXiv](https://img.shields.io/badge/arXiv-2408.11065-b31b1b.svg)](https://arxiv.org/abs/2408.11065)

This repository contains the code and the data used to perform the analysis described in 
[Constantin et al. 2024](https://arxiv.org/abs/2408.11065). If you wish to cite this paper, please
use the following bibtex

```
@ARTICLE{2024arXiv240811065C,
       author = {{Constantin}, Andrei and {Bartlett}, Deaglan and {Desmond}, Harry and {Ferreira}, Pedro G.},
        title = "{Statistical Patterns in the Equations of Physics and the Emergence of a Meta-Law of Nature}",
      journal = {arXiv e-prints},
     keywords = {Physics - Physics and Society, Computer Science - Computation and Language, High Energy Physics - Theory, Physics - Data Analysis, Statistics and Probability, Physics - History and Philosophy of Physics},
         year = 2024,
        month = aug,
          eid = {arXiv:2408.11065},
        pages = {arXiv:2408.11065},
          doi = {10.48550/arXiv.2408.11065},
archivePrefix = {arXiv},
       eprint = {2408.11065},
 primaryClass = {physics.soc-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240811065C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` 

The function sets used for the analysis can be found in the `data/` directory.

To generate random algebraic expressions, one can use the Mathematica notebook `RandomExpressions.nb`.

To perform the implicit inference tests, one can use the script `ili_zipf.py`, which utilises the `ltu-ili`
package ([Ho et al. 2024](https://arxiv.org/abs/2402.05137)). This can be downloaded and installed as follows

```bash
conda create -n ili-torch python=3.10 
conda activate ili-torch
pip install --upgrade pip
pip install -e git+https://github.com/maho3/ltu-ili#egg=ltu-ili
```

