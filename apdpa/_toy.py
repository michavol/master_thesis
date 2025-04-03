import torch
from scipy.special import comb
from torch import nn
from sklearn.preprocessing import PolynomialFeatures

class PolyDecoder(nn.Module):
    """
    Injective polynomial decoder for the latent space.
    """

    def __init__(self, deg, x_dim, lat_dim, debug=False):
        super().__init__()
        # Properties of data and latent space
        self.deg = deg
        self.x_dim = x_dim
        self.lat_dim = lat_dim
        self.debug = debug

        # Compute the total number of polynomial terms
        self.poly_size = self.num_polynomial_terms_of_degree_p(self.deg)

        # Check the implicit dimensionality condition for full column-rankedness
        assert self.x_dim >= self.num_polynomial_terms_of_degree_p(self.deg), (
            "The polynomial degree is too high for the latent dimensionality "
            "to guarantee an injective polynomial decoder."
        )

        # Generate a full-rank coefficient matrix (for injectivity)
        self.coef_matrix = self.random_full_column_rank()

    def num_polynomial_terms_of_degree_p(self, p):
        """
        Compute the number of polynomial terms for a given degree p.
        Uses the combinatorial formula for the number of non-negative
        integer solutions to x1 + x2 + ... + xk = p.
        """
        count = 0
        for r in range(p + 1):
            count += comb(r + self.lat_dim - 1, self.lat_dim - 1)
        return int(count)

    def compute_total_num_polynomial_terms(self):
        """
        Compute the total number of possible terms for polynomials
        up to degree self.deg.
        """
        count = 0
        for p in range(self.deg + 1):
            count += self.num_polynomial_terms_of_degree_p(p)
        return count

    def random_full_column_rank(self):
        """
        Generate an n x p matrix (p <= n) with real entries drawn from
        a normal distribution. With probability 1, it will be full column rank.
        If not, the function regenerates until it is.
        """
        n = self.x_dim
        p = self.poly_size

        while True:
            M = torch.randn(n, p)
            rank_M = torch.linalg.matrix_rank(M)
            if rank_M == p:
                return M

    def compute_decoder_polynomial(self, latent):
        """
        Compute the polynomial features of the latent vector up to degree self.deg.
        Uses scikit-learn's PolynomialFeatures to include interactions.
        """
        assert latent.shape[0] == self.lat_dim, (
            "The latent dimensionality of the sample is incorrect."
        )

        poly = PolynomialFeatures(degree=self.deg, include_bias=True)  
        out = poly.fit_transform(latent.reshape(1, -1)).T  # shape -> (n_features, 1)

        return torch.tensor(out, dtype=torch.float32)

    def forward(self, latent):
        """
        Forward pass of the polynomial decoder.
        1) Compute polynomial terms of latent.
        2) Multiply by the learned coefficient matrix.
        """
        latent_poly = self.compute_decoder_polynomial(latent)
        X = torch.matmul(self.coef_matrix, latent_poly)

        if self.debug:
            print(f"\nShape of the latent vector: {latent.shape}")
            print(latent)
            print(f"\nShape of the polynomial terms: {latent_poly.shape}")
            print(latent_poly)
            print(f"\nShape of coefficients: {self.coef_matrix.shape}")
            print(self.coef_matrix)
            print(f"\nShape of the output: {X.shape}")
            print(X)

        return X
    

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from iscm import data_sampler, graph_utils
np.set_printoptions(precision=3, suppress=True)

class LatentSCM:
    def __init__(
        self,
        adjacency_matrix,
        intervention_dict,
        rng_seed=0,
        noise_variance=1e-5,
        noise_distribution="gaussian"
    ):
        """
        Construct an SCM from a user-supplied weighted adjacency matrix, labeling
        nodes as upstream, downstream, or interventional.

        Parameters
        ----------
        adjacency_matrix : np.ndarray
            A square (n x n) matrix describing the DAG. Nonzero entry [i, j]
            indicates a directed edge i -> j with weight adjacency_matrix[i, j].
        intervention_dict : dict
            Dictionary of potential do-interventions, e.g. {node_idx: do_value, ...}.
            All keys in this dict are considered 'interventional nodes'.
        rng_seed : int
            Seed for the internal RNG.
        noise_variance : float
            Variance of the noise terms used in sampling.
        noise_distribution : str
            Distribution name for data_sampler (e.g. 'gaussian').
        """
        self.adjacency_matrix = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
        self.intervention_dict = intervention_dict.copy()
        self.noise_variance = noise_variance
        self.noise_distribution = noise_distribution
        self.rng = np.random.default_rng(rng_seed)

        # Build the graph object from ds.py
        self.graph = graph_utils.Graph(weight_matrix=self.adjacency_matrix)
        
        # Classify nodes
        self._classify_nodes()

    def _classify_nodes(self):
        """
        Classify nodes as:
         - Interventional (keys of self.intervention_dict)
         - Downstream (descendants of any interventional node)
         - Upstream (all other nodes).
        """
        self.interventional_nodes = set(self.intervention_dict.keys())

        # Use NetworkX to find descendants
        nx_graph = self.graph.get_nx_graph()

        all_descendants = set()
        for intv_node in self.interventional_nodes:
            all_descendants |= nx.descendants(nx_graph, intv_node)

        # Downstream = all_descendants minus any interventional node
        self.downstream_nodes = all_descendants - self.interventional_nodes

        all_nodes = set(range(self.n_nodes))
        self.upstream_nodes = all_nodes - self.interventional_nodes - self.downstream_nodes

    def sample(
        self,
        do_intervention_dict=None,
        sample_size=100,
        standardization="internal",
    ):
        """
        Sample from the SCM with (possibly overridden) do-interventions.
        If do_intervention_dict is None, we sample from the unperturbed system.

        Parameters
        ----------
        do_intervention_dict : dict, optional
            Dictionary of do-interventions to apply. Keys are node indices,
            values are the do-values. If None, no interventions are applied.
        sample_size : int
            Number of samples to draw.
        standardization : str
            Standardization mode for data_sampler.

        Returns
        -------
        np.ndarray
            shape = (sample_size, n_nodes).
        """
        # If user provides no dictionary, sample unperturbed
        if do_intervention_dict is None:
            do_dict = {}
        else:
            do_dict = do_intervention_dict

        samples = data_sampler.sample_linear(
            graph=self.graph,
            do_interventions=do_dict,
            noise_variance=self.noise_variance,
            standardization=standardization,
            sample_size=sample_size,
            noise_distribution=self.noise_distribution,
            rng=self.rng,
        )
        return samples

    def plot_graph(self, title="SCM Graph"):
        """
        Plot the DAG, coloring interventional nodes (red), downstream (blue), upstream (green).
        Each node is labeled by its integer index to match adjacency matrix ordering.
        """
        nx_graph = self.graph.get_nx_graph()

        # Build color map
        color_map = []
        for node in nx_graph.nodes():
            if node in self.interventional_nodes:
                color_map.append("red")
            elif node in self.downstream_nodes:
                color_map.append("blue")
            else:
                color_map.append("green")

        # Create positions and label dict
        pos = nx.spring_layout(nx_graph, seed=42)
        labels = {node: str(node) for node in range(self.n_nodes)}

        plt.figure(figsize=(5, 4))
        nx.draw(
            nx_graph,
            pos,
            node_color=color_map,
            labels=labels,  # ensures each node is labeled with its own index
            with_labels=True
        )

        # Construct legend
        red_patch = plt.Line2D([0], [0], marker='o', color='w',
                               label='Interventional', markerfacecolor='red', markersize=10)
        blue_patch = plt.Line2D([0], [0], marker='o', color='w',
                                label='Downstream', markerfacecolor='blue', markersize=10)
        green_patch = plt.Line2D([0], [0], marker='o', color='w',
                                 label='Upstream', markerfacecolor='green', markersize=10)
        
        plt.legend(handles=[red_patch, blue_patch, green_patch], loc='best')
        plt.title(title)
        plt.show()
