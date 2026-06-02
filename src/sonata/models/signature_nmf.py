from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ..initialization.initialize import EPSILON, initialize_standard_nmf
from ..initialization.methods import _INIT_METHODS
from ..utils import match_signatures_pair, type_checker, value_checker

if TYPE_CHECKING:
    from ..initialization.methods import _Init_methods


class SignatureNMF(ABC):
    """
    Base class for NMF models used in mutational signature analysis.

    SignatureNMF manages the shared fitting loop and common model state.
    Fitted count data are stored in `adata`, learned signatures in
    `asignatures`, and optional objective values in `history`.

    Standard NMF-style models use the default initialization for signatures
    and exposures. Models with additional parameters override `_initialize`.

    Subclasses define the model-specific objective, parameter updates,
    fitting setup, and sample-wise reconstruction errors.
    """

    def __init__(
        self,
        n_signatures: int = 1,
        init_method: _Init_methods = "nndsvd",
        min_iterations: int = 500,
        max_iterations: int = 10000,
        conv_test_freq: int = 10,
        tol: float = 1e-7,
    ):
        """
        Inputs
        ------
        n_signatures: int
            The number of signatures that are assumed to
            have generated the mutation count data.

        init_method: str, default='nndsvd'
            The model parameter initialization method.

        min_iterations: int, default=500
            The minimum number of iterations to perform by the NMF algorithm

        max_iterations: int, default=10000
            The maximum number of iterations to perform by the NMF algorithm

        conv_test_freq: int, default=10
            The frequency at which the algorithm is tested for convergence.
            The objective function value is only computed every 'conv_test_freq'
            many iterations.

        tol: float
            The convergence tolerance. The NMF algorithm is converged
            when the relative change of the objective function is smaller
            than 'tol'.
        """
        value_checker("init_method", init_method, _INIT_METHODS)

        self.n_signatures = n_signatures
        self.init_method = init_method
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.conv_test_freq = conv_test_freq
        self.tol = tol

        # initialize data/fitting dependent attributes
        self.adata = AnnData()
        self.asignatures = AnnData()
        self.history: dict[str, Any] = {}

    @property
    def mutation_types(self) -> list[str]:
        return list(self.adata.var_names)

    @property
    def signature_names(self) -> list[str]:
        return list(self.asignatures.obs_names)

    @property
    def sample_names(self) -> list[str]:
        return list(self.adata.obs_names)

    @property
    def signatures(self) -> pd.DataFrame:
        """
        Extract the mutational signatures as a pandas dataframe.
        """
        return self.asignatures.to_df()

    @property
    def exposures(self) -> pd.DataFrame:
        """
        Extract the signature exposures as a pandas dataframe.
        """
        assert (
            "exposures" in self.adata.obsm
        ), "Learning the sample exposures requires fitting the NMF model."
        exposures_df = pd.DataFrame(
            self.adata.obsm["exposures"],
            index=self.sample_names,
            columns=self.signature_names,
        )
        return exposures_df

    def compute_reconstruction(self) -> None:
        self.adata.obsm["X_reconstructed"] = (
            self.adata.obsm["exposures"] @ self.asignatures.X
        )

    @property
    def data_reconstructed(self) -> pd.DataFrame:
        if "X_reconstructed" not in self.adata.obsm:
            self.compute_reconstruction()

        return pd.DataFrame(
            self.adata.obsm["X_reconstructed"],
            index=self.sample_names,
            columns=self.mutation_types,
        )

    @abstractmethod
    def compute_reconstruction_errors(self) -> None:
        """
        The samplewise reconstruction errors between the data
        and the reconstructed data.
        """

    @property
    def reconstruction_error(self) -> float:
        """
        The total reconstruction error between the data and
        the reconstructed data.
        """
        if "reconstruction_error" not in self.adata.obs:
            self.compute_reconstruction_errors()

        return np.sum(self.adata.obs["reconstruction_error"])

    @property
    @abstractmethod
    def objective(self) -> Literal["minimize", "maximize"]:
        """
        Whether the NMF algorithm minimizes or maximizes its objective
        function.
        """

    @abstractmethod
    def objective_function(self) -> float:
        """
        The objective function to be optimized during fitting.
        """

    def _setup_adata(self, adata: AnnData) -> None:
        """
        Check the type of the input counts and clip them to
        avoid floating point errors.

        Inputs
        ------
        data: AnnData
            The AnnData object with the mutation count matrix.
        """
        type_checker("adata", adata, AnnData)
        self.adata = adata
        self.adata.X = self.adata.X.clip(EPSILON)

    def _initialize(
        self,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the standard signature and exposure parameters.

        Models with additional parameters, such as Cornet, override this
        method.
        """
        init_kwargs = {} if init_kwargs is None else init_kwargs.copy()
        self.asignatures = initialize_standard_nmf(
            self.adata,
            self.n_signatures,
            self.init_method,
            given_parameters,
            **init_kwargs,
        )

    @abstractmethod
    def _setup_fitting_parameters(
        self, fitting_kwargs: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize any additional and parameters required to fit
        the NMF model.
        """

    @abstractmethod
    def _update_parameters(
        self, given_parameters: dict[str, Any] | None = None
    ) -> None:
        """
        Update all model parameters.
        """

    def fit(
        self,
        adata: AnnData,
        given_parameters: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
        fitting_kwargs: dict[str, Any] | None = None,
        history: bool = True,
        verbose: Literal[0, 1] = 0,
        verbosity_freq: int = 1000,
    ) -> SignatureNMF:
        """
        Fit the model parameters. NMF models are expected to handle
        'given_parameters' appropriately.

        Inputs
        ------
        adata: AnnData
            The mutation count matrix as an AnnData object.

        given_parameters: dict, optional
            A priori known parameters. The key is expected to be the parameter
            name.

        init_kwargs: dict, optional
            Keyword arguments to pass to the model parameter initialization, e.g.,
            a seed when a stochastic initialization method is used.

        fitting_kwargs: dict, optional
            Keyword arguments to pass to the initialization of additional fitting
            parameters, e.g., sample-specific loss function weights.

        history: bool, default=True
            If True, the objective function values computed during model training
            will be stored.

        verbose: Literal[0, 1], default=0
            If True, intermediate objective function values obtained during model
            training are printed.

        verbosity_freq: int, default=1000
            The objective function values after every 'verbosity_freq' many
            iterations are printed. Only applies if 'verbose' is set to 1.
        """
        self._setup_adata(adata)
        self._initialize(given_parameters, init_kwargs)
        self._setup_fitting_parameters(fitting_kwargs)
        of_values = [self.objective_function()]
        n_iteration = 0
        converged = False

        while not converged:
            n_iteration += 1

            if verbose and n_iteration % verbosity_freq == 0:
                print(f"iteration: {n_iteration}; objective: {of_values[-1]:.2f}")

            self._update_parameters(given_parameters)

            if n_iteration % self.conv_test_freq == 0:
                prev_of_value = of_values[-1]
                of_values.append(self.objective_function())
                rel_change_nominator = np.abs(prev_of_value - of_values[-1])
                rel_change = rel_change_nominator / np.abs(prev_of_value)
                converged = rel_change < self.tol and n_iteration >= self.min_iterations

            converged |= n_iteration >= self.max_iterations

        if history:
            self.history["objective_function"] = of_values[1:]

        return self

    def reorder(
        self,
        asignatures_other: AnnData,
        metric: str = "cosine",
        keep_names=False,
    ) -> None:
        """
        Reorder the model parameters to match the order of another
        collection of signatures.
        """
        names = self.asignatures.obs_names
        reordered_indices = match_signatures_pair(
            asignatures_other.to_df(), self.asignatures.to_df(), metric=metric
        )
        self.asignatures = self.asignatures[reordered_indices, :].copy()
        self.adata.obsm["exposures"] = self.adata.obsm["exposures"][
            :, reordered_indices
        ]
        if not keep_names:
            self.asignatures.obs_names = names
