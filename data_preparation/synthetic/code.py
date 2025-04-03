import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import inspect
from typing import List, Dict, Callable, Any, Optional, Union, Tuple, Set

# Default noise generation function (Gaussian)
def no_noise(size: int, strength: float) -> np.ndarray:
    """Generates no noise (zeros)."""
    return np.zeros(size)

def gaussian_noise(size: int, strength: float) -> np.ndarray:
    """Generates Gaussian noise."""
    return np.random.normal(loc=0.0, scale=strength, size=size)

# Example alternative noise generation function (Uniform)
def uniform_noise(size: int, strength: float) -> np.ndarray:
    """Generates Uniform noise."""
    limit = np.sqrt(3) * strength
    return np.random.uniform(low=-limit, high=limit, size=size)

NOISE_GENERATORS = {
    'noiseless': no_noise,
    'gaussian': gaussian_noise,
    'uniform': uniform_noise,
}

class StructuralCausalModel:
    """
    Represents and samples from a Structural Causal Model (SCM).

    Handles variable groups (upstream/variant), perfect do-interventions
    (single or combined) with strength per intervention, graph structure constraints,
    plotting, context parameters (e.g., cell lines), and global noise settings.
    """

    def __init__(self,
                 structural_equations: Dict[str, Callable],
                 upstream_vars: Set[str],
                 interventional_vars: Set[str],
                 downstream_vars: Set[str],
                 intervention_definitions: Dict[str, str], # {intervention_type_name: target_variable}
                 noise_strength: float = 1.0,
                 noise_type: str = 'gaussian',
                 default_context_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the Structural Causal Model.

        Args:
            structural_equations: Dictionary mapping variable names (str) to
                their generating functions (Callable). Functions should accept
                parent values as keyword arguments and optionally 'context_params'
                and 'n_samples'.
            upstream_vars: Set of variable names considered upstream (invariant).
            interventional_vars: Set of variable names that can be intervened upon.
            downstream_vars: Set of variable names downstream of interventional vars.
            intervention_definitions: Dictionary mapping intervention type names
                (e.g., "perturb_geneA") to the specific variable name they target
                (e.g., "geneA").
            noise_strength: Global scaling factor for the noise term added to
                each non-intervened variable (default: 1.0).
            noise_type: Type of noise distribution ('gaussian' or 'uniform').
                (default: 'gaussian').
            default_context_params: Optional dictionary of default parameters
                passed to structural equations (simulating e.g., cell lines).
                These can be overridden during sampling.
        """
        self.structural_equations = structural_equations
        self.upstream_vars = upstream_vars
        self.interventional_vars = interventional_vars
        self.downstream_vars = downstream_vars
        self.intervention_definitions = intervention_definitions
        self.noise_strength = noise_strength
        self.default_context_params = default_context_params if default_context_params else {}

        # Combine interventional and downstream into variant variables
        self.variant_vars = self.interventional_vars.union(self.downstream_vars)
        self.all_vars = self.upstream_vars.union(self.variant_vars)

        # Select noise generation function
        if noise_type not in NOISE_GENERATORS:
            raise ValueError(f"Unsupported noise_type: {noise_type}. "
                             f"Available types: {list(NOISE_GENERATORS.keys())}")
        self.noise_generator = NOISE_GENERATORS[noise_type]
        self.noise_type = noise_type

        # Build graph and validate
        self._build_graph()
        self._validate_inputs()
        self._topological_sort = list(nx.topological_sort(self.graph))

    def _get_parents(self, var: str) -> List[str]:
        """Infers parents from the signature of the structural equation."""
        func = self.structural_equations[var]
        sig = inspect.signature(func)
        # Parents are arguments that are not 'n_samples' or 'context_params'
        parents = [p for p in sig.parameters
                   if p not in ['n_samples', 'context_params']]
        return parents

    def _build_graph(self):
        """Builds the NetworkX graph representation from structural equations."""
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.all_vars)
        for var in self.all_vars:
            if var not in self.structural_equations:
                 continue # Roots handled if they appear as parents

            parents = self._get_parents(var)
            for parent in parents:
                if parent not in self.all_vars:
                    raise ValueError(f"Parent '{parent}' of variable '{var}' "
                                     f"is not defined in upstream, interventional, "
                                     f"or downstream variables.")
                self.graph.add_edge(parent, var)

        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            raise ValueError(f"The defined structure contains cycles: {cycles}")


    def _validate_inputs(self):
        """Performs validation checks on the SCM definition."""
        # [Validation checks remain the same as before]
        # 1. Check variable uniqueness across groups
        if not self.upstream_vars.isdisjoint(self.interventional_vars):
            raise ValueError("Overlap detected between upstream and interventional variables.")
        if not self.upstream_vars.isdisjoint(self.downstream_vars):
             raise ValueError("Overlap detected between upstream and downstream variables.")
        if not self.interventional_vars.isdisjoint(self.downstream_vars):
             raise ValueError("Overlap detected between interventional and downstream variables.")

        # 2. Check if all variables with equations are in the defined sets
        for var in self.structural_equations:
            if var not in self.all_vars:
                raise ValueError(f"Variable '{var}' in structural_equations "
                                 f"is not defined in any variable group.")

        # 3. Check intervention definitions
        for int_type, target_var in self.intervention_definitions.items():
            if target_var not in self.interventional_vars:
                raise ValueError(f"Intervention '{int_type}' targets variable "
                                 f"'{target_var}', which is not in the defined "
                                 f"interventional_vars set.")

        # 4. Requirement 3: No upstream descendants of *variant* variables
        for u, v in self.graph.edges():
            if u in self.variant_vars and v in self.upstream_vars:
                raise ValueError(f"Invalid edge: Cannot have an edge from a "
                                 f"variant variable ('{u}') to an upstream "
                                 f"variable ('{v}').")

        # 5. Check if all non-root variables have structural equations
        roots = {node for node, degree in self.graph.in_degree() if degree == 0}
        vars_needing_eqs = self.all_vars - roots
        missing_eqs = vars_needing_eqs - set(self.structural_equations.keys())
        # Allow roots defined only by noise (no parents) not to have explicit equation
        # if they don't appear as parents elsewhere? No, better to require equation for all.
        # Let's ensure all variables have an equation for clarity.
        missing_eqs_all = self.all_vars - set(self.structural_equations.keys())
        if missing_eqs_all:
            raise ValueError(f"Missing structural equations for variables: {missing_eqs_all}")


    def plot_graph(self, ax=None, layout='kamada_kawai', **kwargs):
        """
        Plots the causal graph structure using NetworkX and Matplotlib.

        Args:
            ax: Matplotlib axes object to plot on. If None, creates a new figure.
            layout: NetworkX layout function name (e.g., 'kamada_kawai', 'spring',
                    'circular') or a precomputed layout dictionary.
            **kwargs: Additional keyword arguments passed to nx.draw().
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (8, 6)))
            # Don't set suptitle here, let the main script do it if desired
            # fig.suptitle("SCM Graph Structure")

        node_colors = []
        node_shapes = [] # o for upstream/downstream, s for interventional
        for node in self.graph.nodes():
            if node in self.upstream_vars:
                node_colors.append('lightblue')
                node_shapes.append('o')
            elif node in self.interventional_vars:
                node_colors.append('lightcoral')
                node_shapes.append('s') # Square for interventional
            elif node in self.downstream_vars:
                node_colors.append('lightgreen')
                node_shapes.append('o')
            else:
                node_colors.append('gray') # Should not happen with validation
                node_shapes.append('d')

        # Use a layout algorithm or precomputed positions
        if isinstance(layout, str):
            try:
                layout_func = getattr(nx, f"{layout}_layout")
                pos = layout_func(self.graph)
            except AttributeError:
                 print(f"Warning: Layout '{layout}' not found in NetworkX. Using 'kamada_kawai_layout'.")
                 pos = nx.kamada_kawai_layout(self.graph)
        elif isinstance(layout, dict):
            pos = layout
        else:
            # Default layout
            pos = nx.kamada_kawai_layout(self.graph)


        # Draw nodes distinctly by shape requires drawing nodes separately
        unique_shapes = sorted(list(set(node_shapes))) # Sort for consistent legend order
        handles = []
        labels = {'o': 'Upstream/Downstream', 's': 'Interventional'} # Adjust label map

        node_kwargs = { # Default node drawing options
            'node_size': 700,
            'alpha': 0.9
        }
        node_kwargs.update(kwargs) # Allow overriding via plot_graph call

        for shape in unique_shapes:
            shaped_nodes = [node for node, s in zip(self.graph.nodes(), node_shapes) if s == shape]
            shaped_colors = [color for color, s in zip(node_colors, node_shapes) if s == shape]

            # Map shape character to a color for the legend handle (use average color?)
            # Simpler: use predefined colors for legend handles
            if shape == 's': # Interventional
                 handle_color = 'lightcoral'
                 label_text = 'Interventional'
                 handles.append(plt.Line2D([0], [0], marker=shape, color='w', label=label_text, markersize=10, markerfacecolor=handle_color))
            elif shape == 'o': # Upstream / Downstream need separate handles
                # Check which types are present
                has_upstream = any(n in self.upstream_vars for n in shaped_nodes)
                has_downstream = any(n in self.downstream_vars for n in shaped_nodes)
                if has_upstream:
                     handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Upstream', markersize=10, markerfacecolor='lightblue'))
                if has_downstream:
                     handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Downstream', markersize=10, markerfacecolor='lightgreen'))

            nx.draw_networkx_nodes(self.graph, pos,
                                   nodelist=shaped_nodes,
                                   node_color=shaped_colors,
                                   node_shape=shape, ax=ax, **node_kwargs)


        # Draw edges and labels
        edge_kwargs = {
             'arrowstyle': '->',
             'connectionstyle':'arc3,rad=0.1',
             'node_size': node_kwargs.get('node_size', 700), # Make arrows point near node edge
             'alpha': 0.7
        }
        nx.draw_networkx_edges(self.graph, pos, ax=ax, **edge_kwargs)
        nx.draw_networkx_labels(self.graph, pos, ax=ax)

        # Add legend
        ax.legend(handles=handles, title="Variable Types", loc='best')
        ax.set_title("SCM Graph Structure")
        plt.tight_layout()
        # Don't call plt.show() here, let the calling script manage figures
        # plt.show()


    def sample(self,
               n_samples: int = 1,
               interventions: Optional[Dict[str, float]] = None, # CHANGED: Dict {type: strength}
               context_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Samples data from the SCM, applying optional interventions with specific strengths.

        Args:
            n_samples: Number of samples to generate.
            interventions: Dictionary mapping intervention type names (str) to
                           their desired perturbation strengths (float).
                           Example: {'set_protX_high': 5.0, 'set_protY_low': -2.0}.
                           If None, observational data is sampled.
            context_params: Dictionary of context parameters (e.g., cell line
                            specifics) to potentially override defaults and pass
                            to structural equations.

        Returns:
            Pandas DataFrame containing the sampled data, with columns for
            each variable.
        """
        samples = pd.DataFrame(index=range(n_samples), columns=self._topological_sort, dtype=float)
        intervention_targets = {} # Map from target variable to its specific perturbation strength

        # Process interventions using the new dictionary format
        if interventions:
            if not isinstance(interventions, dict):
                 raise TypeError("The 'interventions' argument must be a dictionary "
                                 "mapping intervention type names to their strengths (float), "
                                 "or None.")

            active_targets = set() # To check for multiple interventions on the same target
            for int_type, strength in interventions.items():
                if int_type not in self.intervention_definitions:
                    raise ValueError(f"Unknown intervention type: '{int_type}'. "
                                     f"Defined types: {list(self.intervention_definitions.keys())}")

                target_var = self.intervention_definitions[int_type]

                if target_var in active_targets:
                     raise ValueError(f"Variable '{target_var}' is targeted by multiple "
                                      f"interventions ({int_type} and previous) in the "
                                      f"same sampling call via the interventions dictionary. "
                                      f"This is ambiguous.")

                if not isinstance(strength, (int, float)):
                     raise TypeError(f"Strength for intervention '{int_type}' must be a "
                                     f"number (float or int), got {type(strength).__name__}.")

                intervention_targets[target_var] = float(strength) # Store target and its strength
                active_targets.add(target_var)

        # Combine default and provided context parameters
        current_context = self.default_context_params.copy()
        if context_params:
            current_context.update(context_params)


        # Generate samples following topological order
        for var in self._topological_sort:
            structural_eq = self.structural_equations.get(var)
            if structural_eq is None:
                 # This case should ideally be caught by validation ensuring all vars have eqs.
                 raise RuntimeError(f"Internal Error: No structural equation found for '{var}' "
                                    f"during sampling, despite passing validation.")

            # Check for intervention ON THIS VARIABLE
            if var in intervention_targets:
                # Apply perfect do-intervention: set value = specific strength for this intervention
                intervention_strength = intervention_targets[var]
                samples[var] = np.full(n_samples, intervention_strength)
            else:
                # Calculate based on structural equation (no intervention on this var)
                parents = self._get_parents(var)
                parent_values = {p: samples[p] for p in parents}

                # Check if context_params or n_samples are needed by the function
                sig = inspect.signature(structural_eq)
                func_args = {}
                if 'context_params' in sig.parameters:
                    func_args['context_params'] = current_context
                if 'n_samples' in sig.parameters:
                    func_args['n_samples'] = n_samples

                # Calculate base value from parents and context
                base_value = structural_eq(**parent_values, **func_args)

                # Add noise (unless it's explicitly handled in the equation)
                noise_already_added = (n_samples > 1 and isinstance(base_value, np.ndarray) and base_value.shape == (n_samples,))

                if not noise_already_added:
                     if not isinstance(base_value, np.ndarray) or base_value.ndim == 0:
                         base_value = np.full(n_samples, base_value)
                     elif base_value.shape != (n_samples,):
                          raise ValueError(f"Structural equation for '{var}' returned unexpected shape "
                                           f"{base_value.shape}, expected ({n_samples},) or scalar.")

                     noise = self.noise_generator(n_samples, self.noise_strength)
                     samples[var] = base_value + noise
                else:
                     samples[var] = base_value # Function handled sampling/noise

        return samples

# ================== EXAMPLE USAGE (Updated) ==================

if __name__ == "__main__":
    # --- Define Structural Equations (same as before) ---
    def eq_gene_a(context_params: Dict, n_samples: int) -> np.ndarray:
        alpha = context_params.get('cell_alpha', 1.0)
        # Return array of size n_samples, assuming noise included internally
        return np.random.poisson(alpha * 5, size=n_samples).astype(float)

    def eq_gene_b(gene_a: np.ndarray) -> np.ndarray:
        return 2.0 * gene_a

    def eq_protein_x(gene_a: np.ndarray, context_params: Dict) -> np.ndarray:
        beta = context_params.get('cell_beta', 0.5)
        # Return deterministic part, noise added by sample()
        return beta * (gene_a**2)

    def eq_protein_y(gene_b: np.ndarray, protein_x: np.ndarray) -> np.ndarray:
        # Return deterministic part
        return 0.8 * gene_b + 1.2 * protein_x

    def eq_phenotype_z(protein_x: np.ndarray, protein_y: np.ndarray) -> np.ndarray:
         # Return deterministic part
         return -0.5 * protein_x + 0.7 * protein_y

    # --- Define SCM Components ---
    structural_equations = {
        'gene_a': eq_gene_a,
        'gene_b': eq_gene_b,
        'protein_x': eq_protein_x,
        'protein_y': eq_protein_y,
        'phenotype_z': eq_phenotype_z
    }

    upstream_vars = {'gene_a', 'gene_b'}
    # Make both protein_x and protein_y interventional for combined example
    interventional_vars = {'protein_x', 'protein_y'}
    downstream_vars = {'phenotype_z'}

    # Define intervention types and their targets
    intervention_definitions = {
        'set_protX_high': 'protein_x',
        'set_protX_low': 'protein_x',
        'set_protY_fixed': 'protein_y',
        'set_protY_other': 'protein_y'
    }

    default_cell_params = {'cell_alpha': 1.0, 'cell_beta': 0.5}
    cell_line_2_params = {'cell_alpha': 1.5, 'cell_beta': 0.8}

    # --- Instantiate the SCM ---
    try:
        scm = StructuralCausalModel(
            structural_equations=structural_equations,
            upstream_vars=upstream_vars,
            interventional_vars=interventional_vars,
            downstream_vars=downstream_vars,
            intervention_definitions=intervention_definitions,
            noise_strength=0.5,
            noise_type='gaussian',
            default_context_params=default_cell_params
        )
        print("SCM Instantiated Successfully.")

        # --- Plot the Graph ---
        print("\nPlotting SCM graph...")
        fig, ax = plt.subplots(figsize=(7, 5)) # Create figure/axes explicitly
        scm.plot_graph(ax=ax) # Pass axes to the method
        plt.show() # Show the plot


        # --- Perform Sampling ---
        n_samples = 1000

        # 1. Sample observational data
        print("\nSampling observational data (default cell line)...")
        obs_data = scm.sample(n_samples=n_samples)
        print(obs_data.head())

        # 2. Sample with single intervention (using the new dictionary format)
        print("\nSampling with single intervention: set_protX_high = 5.0 ...")
        int_data_high = scm.sample(
            n_samples=n_samples,
            interventions={'set_protX_high': 5.0} # Dict format now required
        )
        print(int_data_high.head())
        print(f"Mean Protein X (Intervention High): {int_data_high['protein_x'].mean():.2f}") # Should be 5.0
        print(f"Mean Phenotype Z (Intervention High): {int_data_high['phenotype_z'].mean():.2f}")

        # 3. Sample with combined interventions with *different* strengths
        print("\nSampling with COMBINED interventions: set_protX_low = -2.0 AND set_protY_fixed = 10.0 ...")
        combined_int_data = scm.sample(
            n_samples=n_samples,
            interventions={
                'set_protX_low': -2.0,  # do(protein_x = -2.0)
                'set_protY_fixed': 10.0  # do(protein_y = 10.0)
            },
            context_params=cell_line_2_params # Can still combine with context change
        )
        print(combined_int_data.head())
        print(f"Mean Protein X (Combined Int): {combined_int_data['protein_x'].mean():.2f}") # Should be -2.0
        print(f"Mean Protein Y (Combined Int): {combined_int_data['protein_y'].mean():.2f}") # Should be 10.0
        print(f"Mean Phenotype Z (Combined Int, Cell 2): {combined_int_data['phenotype_z'].mean():.2f}")

        # 4. Example: Combined interventions with the *same* strength
        print("\nSampling with COMBINED interventions: set_protX_high = 4.0 AND set_protY_other = 4.0 ...")
        combined_int_data_same = scm.sample(
            n_samples=n_samples,
            interventions={
                'set_protX_high': 4.0,
                'set_protY_other': 4.0
            }
        )
        print(combined_int_data_same.head())
        print(f"Mean Protein X (Combined Int Same): {combined_int_data_same['protein_x'].mean():.2f}") # Should be 4.0
        print(f"Mean Protein Y (Combined Int Same): {combined_int_data_same['protein_y'].mean():.2f}") # Should be 4.0
        print(f"Mean Phenotype Z (Combined Int Same): {combined_int_data_same['phenotype_z'].mean():.2f}")


    except (ValueError, TypeError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")