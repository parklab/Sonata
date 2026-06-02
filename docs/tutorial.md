# Sonata Tutorial

This tutorial walks through the main Sonata workflow for mutational signature analysis: loading count data, fitting NMF, visualizing signatures and exposures, fixing known signatures, and fitting Cornet.

The runnable notebook version is available at [tutorial.ipynb](tutorial.ipynb). It contains the same analysis and can be used to regenerate the PNG figures in `docs/images/`.

```python
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sonata as so

data_dir = Path("data")
```

## 1. Data

Load the breast cancer SBS count matrix and sample labels. Rows of the `AnnData` object are samples, and columns are mutation types.

```python
counts = pd.read_csv(data_dir / "hrdetect_counts_training.csv", index_col=0).T
labels = pd.read_csv(data_dir / "hrdetect_labels_training.csv", index_col=0)

adata = ad.AnnData(counts)
adata.obs = adata.obs.join(labels)
adata
```

Inspect the count matrix.

```python
adata.to_df().head()
```

Load the COSMIC SBS catalog as reference signatures.

```python
catalog_sbs = pd.read_csv(
    data_dir / "COSMIC_v3.3.1_SBS_GRCh38.csv",
    index_col=0,
).T
catalog_sbs.head()
```

## 2. Fit NMF

Fit NMF with six signatures and keep the objective history so convergence can be inspected.

```python
model = so.models.NMF(n_signatures=6, init_method="random")
model.fit(
    adata.copy(),
    init_kwargs={"seed": 42},
    history=True
)
```

After fitting, learned signatures and exposures are available as pandas data frames.

```python
model.signatures.head()
```

```python
model.exposures.head()
```

The reconstruction error summarizes how well the fitted model reconstructs the observed count matrix.

```python
model.reconstruction_error
```

## 3. Tutorial Figures

### 3.1 Model Assessment

Plot the objective history to inspect convergence.

```python
ax = so.pl.history(
    model.history["objective_function"],
    conv_test_freq=model.conv_test_freq,
)
```

![NMF objective history](images/nmf-history.png)

Plot the learned signatures as mutational profiles.

```python
axes = so.pl.barplot(model.asignatures)
```

![NMF signatures](images/nmf-signatures.png)

Plot sample exposures as a stacked bar chart.

```python
fig, ax = plt.subplots(figsize=(18, 3))
ax = so.pl.stacked_barplot(
    model.exposures,
    reorder_dimensions=True,
    annotate_obs=False,
    title="NMF signature exposures",
    ax=ax,
)
```

![NMF exposures](images/nmf-exposures.png)

Reduce the dimensionality of the exposures with UMAP and plot the result.

```python
palette = ["#0072b2", "#d55e00"]

so.tl.reduce_dimension(
    model.adata,
    basis="exposures",
    method="umap",
    n_components=2,
    random_state=42,
)
ax = so.pl.embedding(
    model.adata,
    basis="umap",
    hue=adata.obs["hrd_label"],
    palette=palette,
)
```

![UMAP of NMF exposures](images/nmf-exposure-umap.png)

Scatter plots can also be customized by storing colors in the `AnnData` object and passing annotation labels explicitly. By default, Sonata adjusts annotations to reduce overlap.

```python
rng = np.random.default_rng(42)
my_colors = ["cornflowerblue", "silver"]
model.adata.obs["my_color"] = rng.choice(my_colors, adata.n_obs)

special_samples = rng.choice(adata.obs_names, 15, replace=False)
annotations = [
    sample if sample in special_samples else ""
    for sample in adata.obs_names
]

ax = so.pl.embedding(
    model.adata,
    "umap",
    color="my_color",
    annotations=annotations,
    adjust_kwargs={"arrowprops": dict(arrowstyle="-", color="k")},
)
```

![Customized NMF embedding](images/nmf-custom-embedding.png)

### 3.2 Additional Visualization Functionality

Plot a sample-level annotation against the HRDetect score.

```python
ax = so.pl.scatter(model.adata, x="hrd_label", y="hrdetect_score")
```

![HRD label and HRDetect score](images/nmf-hrd-scatter.png)

Compute and plot correlations between samples based on their NMF exposures.

```python
so.tl.correlation(model.adata, basis="exposures")
grid = so.pl.correlation(model.adata, figsize=(5, 5))
```

![Sample correlation based on NMF exposures](images/nmf-sample-correlation.png)

The same correlation plotting function can be used for learned signatures.

```python
model.asignatures.obsp["correlation"] = so.tl.correlation_numpy(model.adata.obsm["exposures"].T)
grid = so.pl.correlation(
    model.asignatures,
    annot=True,
    figsize=(4, 4),
)
```

![Correlation between NMF signatures](images/nmf-signature-correlation.png)

## 4. Fixing Known Signatures

Known signatures can be fixed during fitting by passing them through `given_parameters`.

```python
given_asignatures = ad.AnnData(catalog_sbs.loc[["SBS1", "SBS2"]])

model_fixed = so.models.NMF(n_signatures=4)
model_fixed.fit(
    adata.copy(),
    given_parameters={"asignatures": given_asignatures},
    init_kwargs={"seed": 42},
)
model_fixed.signatures.head()
```

Plot the fitted signatures after fixing the known COSMIC signatures.

```python
axes = so.pl.barplot(model_fixed.asignatures)
```

![NMF signatures with fixed known signatures](images/nmf-fixed-signatures.png)

## 5. Cornet

Cornet models exposures through sample offsets, signature offsets, and joint sample/signature embeddings.

```python
bdata = adata.copy()
bdata.X = adata.X / adata.X.sum(axis=1, keepdims=True) * 960
```

```python
cornet_model = so.models.Cornet(
    n_signatures=6,
    init_method="random",
    max_iterations=3000
)
cornet_model.fit(
    bdata,
    init_kwargs={"seed": 42},
    verbose=True,
)
```

Plot the Cornet signatures.

```python
axes = so.pl.barplot(cornet_model.asignatures)
```

![Cornet signatures](images/cornet-signatures.png)

Cornet stores signature offsets in `.obs["offsets"]`.

```python
cornet_model.asignatures.obs["offsets"]
```

Reduce the dimensionality of sample and signature embeddings jointly with UMAP. Signature points are drawn on top of sample points using explicit z-order annotations, and signature labels are passed to `so.pl.embedding_multiple`.

```python
cornet_model.adata.obs["color_embeddings"] = np.where(
    cornet_model.adata.obs["hrd_label"].astype(bool),
    "#d55e00",
    "#0072b2",
)
cornet_model.adata.obs["zorder_embeddings"] = 1
cornet_model.asignatures.obs["color_embeddings"] = "black"
cornet_model.asignatures.obs["zorder_embeddings"] = 2

so.tl.reduce_dimension_multiple(
    [cornet_model.adata, cornet_model.asignatures],
    basis="embeddings",
    method="umap",
    n_components=2,
    random_state=42,
)

fig, ax = plt.subplots(figsize=(4, 4))
ax = so.pl.embedding_multiple(
    [cornet_model.asignatures, cornet_model.adata],
    basis="umap",
    color="color_embeddings",
    zorder="zorder_embeddings",
    annotations=cornet_model.signature_names,
    ax=ax,
)
```

![Joint UMAP of Cornet embeddings](images/cornet-embeddings-umap.png)

## 6. Choosing a Number of Signatures

This simple sweep is a starting point for model selection. For publication-quality analyses, repeat fits across seeds and compare learned signatures against domain knowledge and known catalogs.

```python
results = []

for n_signatures in range(2, 11):
    candidate = so.models.NMF(
        n_signatures=n_signatures,
        init_method="random",
    )
    candidate.fit(adata.copy(), init_kwargs={"seed": 42})
    results.append(
        {
            "n_signatures": n_signatures,
            "reconstruction_error": candidate.reconstruction_error,
        }
    )

results = pd.DataFrame(results)
```

Plot reconstruction error across the candidate values of `n_signatures`.

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(results["n_signatures"], results["reconstruction_error"], marker="o")
ax.set_xlabel("Number of signatures")
ax.set_ylabel("Reconstruction error")
```

![Reconstruction error across candidate numbers of signatures](images/model-selection-reconstruction-error.png)
