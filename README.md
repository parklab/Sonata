# Sonata

[![Python versions supported][python-image]][python-url]
[![License][license-image]][license-url]
[![Code style][style-image]][style-url]

[python-image]: https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue.svg
[python-url]: https://github.com/parklab/Sonata
[license-image]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/parklab/Sonata/blob/main/LICENSE
[style-image]: https://img.shields.io/badge/code%20style-black-000000.svg
[style-url]: https://github.com/psf/black

Sonata is a Python toolkit for fitting and analyzing mutational signatures. It
fits signatures and exposures in
[AnnData](https://anndata.readthedocs.io/en/latest/) objects and provides
analysis and plotting APIs for signature workflows.

## Installation

```bash
pip install sonata-tools
```

The package is installed as `sonata-tools` and imported as `sonata`.

## Quickstart

```python
import sonata as so

model = so.models.NMF(n_signatures=6)
model.fit(adata)

so.pl.barplot(model.asignatures)
so.pl.stacked_barplot(model.exposures)

so.tl.reduce_dimension(
    model.adata,
    basis="exposures",
    method="umap",
)
so.pl.embedding(model.adata, basis="umap")
```

## Data Format

Sonata expects mutation counts in an `AnnData` object:

- `adata.X`: count matrix with shape `n_samples x n_mutation_types`.
- `adata.obs`: optional sample annotations.
- `adata.var`: optional mutation-type annotations.

After fitting, the model stores learned signatures in `model.asignatures` and
sample exposures in `model.adata.obsm["exposures"]`.

## Documentation

For a complete workflow covering data preparation, NMF, visualization,
fixed signatures, Cornet, and simple model selection, see the
[Markdown tutorial][tutorial-md]. A runnable notebook with the same analysis and
figure-generation code is available at [docs/tutorial.ipynb][tutorial-ipynb].

## Models

Sonata currently exposes three algorithms:

- `so.models.NMF`: NMF with the generalized Kullback-Leibler divergence.
- `so.models.MvNMF`: minimum-volume NMF.
- `so.models.Cornet`: correlated NMF with joint sample and signature embeddings.

## License

MIT

[tutorial-md]: https://github.com/parklab/Sonata/blob/main/docs/tutorial.md
[tutorial-ipynb]: https://github.com/parklab/Sonata/blob/main/docs/tutorial.ipynb
