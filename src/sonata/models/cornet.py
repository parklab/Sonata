from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..initialization.initialize import EPSILON, initialize_cornet
from . import _utils_cornet
from ._utils_nmf import samplewise_kl_divergence, update_W
from .signature_nmf import SignatureNMF

if TYPE_CHECKING:
    from ..initialization.methods import _Init_methods


class Cornet(SignatureNMF):
    """
    Deterministic batch correlated NMF.

    Cornet refactors the exposure matrix into signature and sample offsets
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
        restructured and determined by the signature & sample offsets and
        embeddings.
        """
        self.adata.obsm["exposures"] = _utils_cornet.compute_exposures(
            self.asignatures.obs["offsets"].values,
            self.adata.obs["offsets"].values,
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
        return _utils_cornet.elbo_cornet(
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
        Initialize the signature matrix, the signature and sample offsets,
        the signature and sample embeddings, and the variance.

        Parameters
        ----------
        given_parameters: dict, default=None
            A priori known parameters / parameters to fix during model training.
            Allowed keys: 'asignatures', 'signature_offsets', 'sample_offsets',
            'signature_embeddings', 'sample_embeddings'. The values have to
            have the appropriate shape. If 'asignatures' is not None, it is
            expected to be an AnnData object.

        init_kwargs : dict
            Any further keyword arguments to pass to the initialization method
            of the signatures. This includes, for example, a possible 'seed'
            keyword argument for all stochastic initialization methods.
        """
        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.asignatures, self.variance = initialize_cornet(
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
        """
        Compute the auxiliary matrix used by the Cornet parameter updates.
        """
        return _utils_cornet.compute_aux(
            self.adata.X, self.asignatures.X, self.adata.obsm["exposures"]
        )

    def update_sample_offsets(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "sample_offsets" not in given_parameters:
            self.adata.obs["offsets"] = _utils_cornet.update_sample_offsets(
                self.adata.X,
                self.asignatures.obs["offsets"].values,
                self.asignatures.obsm["embeddings"],
                self.adata.obsm["embeddings"],
            )

    def update_signature_offsets(
        self, aux: np.ndarray, given_parameters: dict[str, Any] | None = None
    ) -> None:
        if given_parameters is None:
            given_parameters = {}

        if "signature_offsets" not in given_parameters:
            self.asignatures.obs["offsets"] = _utils_cornet.update_signature_offsets(
                aux,
                self.adata.obs["offsets"].values,
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
            self.asignatures.obsm["embeddings"][k, :] = _utils_cornet.update_embedding(
                embedding_init,
                self.adata.obsm["embeddings"],
                self.asignatures.obs["offsets"][k],
                self.adata.obs["offsets"].values,
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
            self.adata.obsm["embeddings"][d, :] = _utils_cornet.update_embedding(
                embedding_init,
                self.asignatures.obsm["embeddings"],
                self.adata.obs["offsets"][d],
                self.asignatures.obs["offsets"].values,
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

        self.update_sample_offsets(given_parameters)
        self.compute_exposures()
        aux = self._compute_aux()
        self.update_signature_offsets(aux, given_parameters)
        self.update_embeddings(aux, given_parameters)
        self.update_variance(given_parameters)
        self.update_signatures(given_parameters)
