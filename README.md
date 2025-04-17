# cvxstatarb

This code repository accompanies two papers:

- [Finding Moving-Band Statistical Arbitrages via Convex-Concave Optimization](https://web.stanford.edu/~boyd/papers/cvx_ccv_stat_arb.html)
- [A Markowitz Approach to Managing a Dynamic Basket of Moving-Band Statistical Arbitrages](https://web.stanford.edu/~boyd/papers/portfolio_of_SAs.html)

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock).

## Jupyter

We install [JupyterLab](https://jupyter.org) on fly within the aforementioned
virtual environment. Executing

```bash
make jupyter
```

will install and start the jupyter lab.

## Citation

If you want to reference our work in your research, please consider using the following BibTeX for the citations:

```BibTeX
@article{johansson2024finding,
  author    = {K. Johansson and T. Schmelzer and S. Boyd},
  title     = {Finding moving-band statistical arbitrages via convex--concave optimization},
  journal   = {Optimization and Engineering},
  year      = {2024},
  doi       = {10.1007/s11081-024-09933-0},
}
```
and
```BibTeX
@article{johansson2024basket,
  author       = {K. Johansson and T. Schmelzer and S. Boyd},
  title        = {A Markowitz Approach to Managing a Dynamic Basket of Moving-Band Statistical Arbitrages},
  journal      = {arXiv preprint arXiv:2412.02660},
  year         = {2024},
  doi          = {10.48550/arXiv.2412.02660}
}
```
