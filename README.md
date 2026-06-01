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

Sonata is a Python toolkit for mutational signature analysis with non-negative
matrix factorization (NMF). It stores count matrices and learned signatures in
[AnnData](https://anndata.readthedocs.io/en/latest/) objects and provides model
fitting, analysis helpers, and plotting functions for signature workflows.

## Installation

```bash
pip install sonata-learn
```

## Quickstart

```python
import anndata as ad
import pandas as pd
import sonata as so

counts = pd.read_csv("data/hrdetect_counts_training.csv", index_col=0).T
adata = ad.AnnData(counts)

model = so.models.KLNMF(n_signatures=6)
model.fit(adata.copy())

so.pl.barplot(model.asignatures)
so.pl.stacked_barplot(model.exposures, annotate_obs=False)

so.tl.reduce_dimension(
    model.adata,
    basis="exposures",
    method="umap",
    random_state=42,
)
so.pl.embedding(model.adata, basis="umap")
```

## Tutorial

For a complete workflow covering data preparation, KL-NMF, visualization,
fixed signatures, CorrNMF, and simple model selection, see the
[Markdown tutorial](docs/tutorial.md). A runnable notebook with the same
analysis and figure-generation code is available at
[docs/tutorial.ipynb](docs/tutorial.ipynb).

## Models

Sonata currently exposes three NMF models:

- `so.models.KLNMF`: NMF with the generalized Kullback-Leibler divergence.
- `so.models.MvNMF`: minimum-volume NMF.
- `so.models.CorrNMF`: correlated NMF with sample and signature embeddings.

## License

MIT
