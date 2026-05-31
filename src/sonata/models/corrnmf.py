from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform

from .. import plot as pl
from .. import tools as tl
from ..initialization.initialize import EPSILON, initialize_corrnmf
from ..utils import value_checker
from . import _utils_corrnmf
from ._utils_klnmf import samplewise_kl_divergence, update_W
from .signature_nmf import SignatureNMF

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..initialization.methods import _Init_methods
    from .signature_nmf import _Dim_reduction_methods


class CorrNMF(SignatureNMF):
    """
    Deterministic batch correlated NMF.

    CorrNMF refactors the exposure matrix into signature and sample scalings
    plus signature and sample embeddings, which lets the model learn a
    correlation structure between signatures and samples.

    Reference
    ---------
    JW Paisley, DM Blei, MI Jordan: Bayesian Nonnegative Matrix Factorization
    with Stochastic Variational Inference, 2014
    """

    def __init__(
        self,
        n_signatures: int = 1,
        init_method: _Init_methods = "nndsvd",
        dim_embeddings: int | None = None,
        min_iterations: int = 500,
        max_iterations: int = 10000,
        conv_test_freq: int = 10,
        tol: float = 1e-7,
    ):
        """
        Input:
        ------
        dim_embeddings: int
            The assumed dimension of the signature and sample embeddings.
            Should be smaller or equal to the number of signatures as a dimension
            equal to the number of signatures covers the case of independent
            signatures. The smaller the embedding dimension, the stronger the
            enforced correlation structure on both signatures and samples.
        """
        super().__init__(
            n_signatures,
            init_method,
            min_iterations,
            max_iterations,
            conv_test_freq,
            tol,
        )
        if dim_embeddings is None:
            dim_embeddings = n_signatures

        self.dim_embeddings = dim_embeddings
        self.variance = 1.0

    def compute_exposures(self) -> None:
        """
        In contrast to the classical NMF framework, the exposure matrix is
        restructured and determined by the signature & sample biases and
        embeddings.
        """
        self.adata.obsm["exposures"] = _utils_corrnmf.compute_exposures(
            self.asignatures.obs["scalings"].values,
            self.adata.obs["scalings"].values,
            self.asignatures.obsm["embeddings"],
            self.adata.obsm["embeddings"],
        )

    def compute_reconstruction_errors(self):
        self.compute_exposures()
        errors = samplewise_kl_divergence(
            self.adata.X.T, self.asignatures.X.T, self.adata.obsm["exposures"].T
        )
        self.adata.obs["reconstruction_error"] = errors

    def objective_function(self) -> float:
        """
        The evidence lower bound (ELBO)
        """
        return _utils_corrnmf.elbo_corrnmf(
            self.adata.X,
            self.asignatures.X,
            self.adata.obsm["exposures"],
            self.asignatures.obsm["embeddings"],
            self.adata.obsm["embeddings"],
            self.variance,
        )

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"

    def _initialize(
        self,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the signature matrix, the signature and sample scalings,
        the signature and sample embeddings, and the variance.

        Parameters
        ----------
        given_parameters: dict, default=None
            A priori known parameters / parameters to fix during model training.
            Allowed keys: 'asignatures', 'signature_scalings', 'sample_scalings',
            'signature_embeddings', 'sample_embeddings'. The values have to
            have the appropriate shape. If 'asignatures' is not None, it is
            expected to be an AnnData object.

        init_kwargs : dict
            Any further keyword arguments to pass to the initialization method
            of the signatures. This includes, for example, a possible 'seed'
            keyword argument for all stochastic initialization methods.
        """
        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.asignatures, self.variance = initialize_corrnmf(
            self.adata,
            self.n_signatures,
            self.dim_embeddings,
            self.init_method,
            given_parameters,
            **init_kwargs,
        )
        self.compute_exposures()

    def _setup_fitting_parameters(
        self, fitting_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        No additional fitting parameters implemented so far.
        """
        return

    def _compute_aux(self) -> np.ndarray:
        return _utils_corrnmf.compute_aux(
            self.adata.X, self.asignatures.X, self.adata.obsm["exposures"]
        )

    def update_sample_scalings(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "sample_scalings" not in given_parameters:
            self.adata.obs["scalings"] = _utils_corrnmf.update_sample_scalings(
                self.adata.X,
                self.asignatures.obs["scalings"].values,
                self.asignatures.obsm["embeddings"],
                self.adata.obsm["embeddings"],
            )

    def update_signature_scalings(
        self, aux: np.ndarray, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "signature_scalings" not in given_parameters:
            self.asignatures.obs["scalings"] = _utils_corrnmf.update_signature_scalings(
                aux,
                self.adata.obs["scalings"].values,
                self.asignatures.obsm["embeddings"],
                self.adata.obsm["embeddings"],
            )

    def update_variance(self, given_parameters: dict[str, Any] | None = None) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "variance" not in given_parameters:
            embeddings = np.concatenate(
                [self.asignatures.obsm["embeddings"], self.adata.obsm["embeddings"]]
            )
            variance = np.mean(embeddings**2)
            self.variance = np.clip(variance, EPSILON, None)

    def update_signatures(self, given_parameters: dict[str, Any] | None = None) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "asignatures" in given_parameters:
            n_given_signatures = given_parameters["asignatures"].n_obs
        else:
            n_given_signatures = 0

        W = update_W(
            self.adata.X.T,
            self.asignatures.X.T,
            self.adata.obsm["exposures"].T,
            n_given_signatures=n_given_signatures,
        )
        self.asignatures.X = W.T

    def update_signature_embeddings(self, aux: np.ndarray) -> None:
        r"""
        Update all signature embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method.

        aux: np.ndarray
            aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        outer_prods_sample_embeddings = np.einsum(
            "Dm,Dn->Dmn",
            self.adata.obsm["embeddings"],
            self.adata.obsm["embeddings"],
        )
        for k, aux_row in enumerate(aux):
            embedding_init = self.asignatures.obsm["embeddings"][k, :]
            self.asignatures.obsm["embeddings"][k, :] = _utils_corrnmf.update_embedding(
                embedding_init,
                self.adata.obsm["embeddings"],
                self.asignatures.obs["scalings"][k],
                self.adata.obs["scalings"].values,
                self.variance,
                aux_row,
                outer_prods_sample_embeddings,
            )

    def update_sample_embeddings(self, aux: np.ndarray) -> None:
        r"""
        Update all sample embeddings by optimizing
        the surrogate objective function using scipy.optimize.minimize
        with the 'Newton-CG' method (strictly convex for each embedding).

        aux: np.ndarray
            aux_kd = \sum_v X_vd * p_vkd
            is used for updating the signatures and the sample embeddidngs.
        """
        outer_prods_signature_embeddings = np.einsum(
            "Km,Kn->Kmn",
            self.asignatures.obsm["embeddings"],
            self.asignatures.obsm["embeddings"],
        )
        for d, aux_col in enumerate(aux.T):
            embedding_init = self.adata.obsm["embeddings"][d, :]
            self.adata.obsm["embeddings"][d, :] = _utils_corrnmf.update_embedding(
                embedding_init,
                self.asignatures.obsm["embeddings"],
                self.adata.obs["scalings"][d],
                self.asignatures.obs["scalings"].values,
                self.variance,
                aux_col,
                outer_prods_signature_embeddings,
                options={"maxiter": 3},
            )

    def update_embeddings(
        self,
        aux: np.ndarray,
        given_parameters: dict[str, Any] | None = None,
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "signature_embeddings" not in given_parameters:
            self.update_signature_embeddings(aux)

        if "sample_embeddings" not in given_parameters:
            self.update_sample_embeddings(aux)

    def _update_parameters(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        self.update_sample_scalings(given_parameters)
        self.compute_exposures()
        aux = self._compute_aux()
        self.update_signature_scalings(aux, given_parameters)
        self.update_embeddings(aux, given_parameters)
        self.update_variance(given_parameters)
        self.update_signatures(given_parameters)

    def compute_correlation_scaled(
        self, data: Literal["samples", "signatures"] = "signatures"
    ) -> None:
        """
        Compute the signature or sample correlation based on the
        scaled exposures and store it in the respective anndata object.
        """
        value_checker("data", data, ["samples", "signatures"])
        assert "embeddings" in self.adata.obsm, (
            "Computing the sample or signature correlation "
            "requires fitting the CorrNMF model."
        )

        if data == "samples":
            vectors = self.adata.obsm["embeddings"]
        else:
            vectors = self.asignatures.obsm["embeddings"]

        norms = np.sqrt(np.sum(vectors**2, axis=1))
        n_vectors = len(norms)
        corr_vector = np.array(
            [
                np.dot(v1, v2) / (norms[i1] * norms[i1 + i2 + 1])
                for i1, v1 in enumerate(vectors)
                for i2, v2 in enumerate(vectors[(i1 + 1) :, :])
            ]
        )
        correlation = squareform(corr_vector) + np.identity(n_vectors)

        if data == "samples":
            self.adata.obsp["X_correlation"] = correlation
        else:
            self.asignatures.obsp["correlation"] = correlation

    def plot_embeddings(
        self,
        method: _Dim_reduction_methods = "umap",
        n_components: int = 2,
        dimensions: tuple[int, int] = (0, 1),
        color: str | None = None,
        zorder: str | None = None,
        annotations: Iterable[str] | None = None,
        outfile: str | None = None,
        **kwargs,
    ) -> Axes:
        adatas = [self.asignatures, self.adata]
        tl.reduce_dimension_multiple(
            adatas=adatas,
            basis="embeddings",
            method=method,
            n_components=n_components,
            **kwargs,
        )
        if self.dim_embeddings <= 2:
            warnings.warn(
                f"The embedding dimension is {self.dim_embeddings}. "
                "The embeddings are plotted without an additional "
                "dimensionality reduction.",
                UserWarning,
            )
            basis = "embeddings"
        else:
            basis = method

        if color is None:
            color = "color_embeddings"
            self.asignatures.obs[color] = self.n_signatures * ["black"]
            self.adata.obs[color] = self.adata.n_obs * ["#1f77b4"]  # default blue

        if zorder is None:
            zorder = "zorder_embeddings"
            self.asignatures.obs[zorder] = self.n_signatures * [2]
            self.adata.obs[zorder] = self.adata.n_obs * [1]

        if annotations is None:
            annotations = self.signature_names

        ax = pl.embedding_multiple(
            adatas=adatas,
            basis=basis,
            dimensions=dimensions,
            color=color,
            zorder=zorder,
            annotations=annotations,
            **kwargs,
        )
        if outfile is not None:
            plt.savefig(outfile, bbox_inches="tight")

        return ax
