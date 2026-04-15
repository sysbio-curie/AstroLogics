import glob
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def parse_bnet_file(filepath: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse a .bnet file and extract regulatory relationships."""
    edges: list[dict[str, str]] = []
    nodes: set[str] = set()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Parse format: Target, Expression
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue

            target = parts[0].strip()
            expression = parts[1].strip()
            nodes.add(target)

            # Extract all regulators (node names) from the expression
            regulators = set(re.findall(r"\b[A-Za-z0-9]+\b", expression))

            # Determine activation/inhibition for each regulator
            nodes.update(regulators)
            for regulator in regulators:
                # Check if regulator is negated (preceded by !)
                is_inhibitory = bool(re.search(rf"!{re.escape(regulator)}", expression))
                is_activating = (
                    bool(re.search(rf"(?<!!)({re.escape(regulator)})(?!\w)", expression))
                    and not is_inhibitory
                )

                edges.append(
                    {
                        "source": regulator,
                        "target": target,
                        "type": "inhibition" if is_inhibitory else "activation" if is_activating else "unknown",
                    }
                )

    return pd.DataFrame(edges), sorted(nodes)


def create_feature_matrix(edges_df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """Create an interaction matrix with nodes as rows/columns and interaction types as values."""
    interaction_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    for _, row in edges_df.iterrows():
        source = row["source"]
        target = row["target"]
        interaction_type = row["type"]

        if interaction_type == "activation":
            interaction_matrix.loc[source, target] = 1
        elif interaction_type == "inhibition":
            interaction_matrix.loc[source, target] = -1
        else:
            interaction_matrix.loc[source, target] = 0

    return interaction_matrix


def flatten_adjacency(matrix: np.ndarray, exclude_diagonal: bool = True) -> np.ndarray:
    """Flatten a directed adjacency matrix into a 1D vector."""
    n = matrix.shape[0]
    if exclude_diagonal:
        mask = ~np.eye(n, dtype=bool)
        return matrix[mask].flatten()
    return matrix.flatten()


def hamming_distance(v1: np.ndarray, v2: np.ndarray) -> int:
    """Compute Hamming distance: number of positions where v1 != v2."""
    return int(np.sum(v1 != v2))


def hamming_distance_normalized(v1: np.ndarray, v2: np.ndarray) -> float:
    """Normalized Hamming distance in [0, 1]."""
    return float(np.sum(v1 != v2) / len(v1))


def weighted_hamming_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Weighted Hamming for ternary vectors {-1, 0, 1}."""
    return float(np.sum(np.abs(v1 - v2)))


def pairwise_hamming_matrix(
    networks: list[np.ndarray],
    exclude_diagonal: bool = True,
    method: str = "hamming",
) -> np.ndarray:
    """Compute pairwise distance matrix for a collection of directed networks."""
    distance_fn = {
        "hamming": hamming_distance,
        "normalized": hamming_distance_normalized,
        "weighted": weighted_hamming_distance,
    }[method]

    vectors = [flatten_adjacency(net, exclude_diagonal) for net in networks]
    m = len(vectors)
    dist_matrix = np.zeros((m, m))

    for i, j in combinations(range(m), 2):
        d = distance_fn(vectors[i], vectors[j])
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

    return dist_matrix


class NetworkSeries:
    """Container for storing flattened network vectors with metadata."""

    def __init__(self):
        self.networks = []
        self.filenames = []
        self.nodes_list = []
        self.flattened_vectors = []
        self.feature_matrices = []

    def add_network(self, filename, edges_df, nodes, feature_matrix, flattened_vector):
        """Add a network to the series."""
        self.filenames.append(filename)
        self.nodes_list.append(nodes)
        self.feature_matrices.append(feature_matrix)
        self.flattened_vectors.append(flattened_vector)
        self.networks.append(
            {
                "filename": filename,
                "nodes": nodes,
                "feature_matrix": feature_matrix,
                "flattened_vector": flattened_vector,
            }
        )

    def get_common_nodes(self) -> list[str]:
        """Get the union of all nodes across all networks."""
        all_nodes = set()
        for nodes in self.nodes_list:
            all_nodes.update(nodes)
        return sorted(all_nodes)

    def normalize_to_common_nodes(self):
        """Normalize all feature matrices to use the same set of nodes."""
        common_nodes = self.get_common_nodes()
        normalized_matrices = []
        normalized_vectors = []

        for feature_matrix in self.feature_matrices:
            normalized_matrix = feature_matrix.reindex(
                index=common_nodes,
                columns=common_nodes,
                fill_value=0,
            )
            normalized_matrices.append(normalized_matrix)
            normalized_vectors.append(normalized_matrix.values.flatten())

        self.feature_matrices = normalized_matrices
        self.flattened_vectors = normalized_vectors

        print(f"Normalized {len(self.networks)} networks to {len(common_nodes)} common nodes")
        print(f"Vector length: {len(normalized_vectors[0])}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert flattened vectors to a pandas DataFrame for analysis."""
        if not self.flattened_vectors:
            return pd.DataFrame()

        vector_length = len(self.flattened_vectors[0])
        return pd.DataFrame(
            self.flattened_vectors,
            columns=[f"edge_{i}" for i in range(vector_length)],
            index=[Path(f).stem for f in self.filenames],
        )

    def validate_uniform_size(self) -> bool:
        """Validate that all flattened vectors have the same size."""
        if not self.flattened_vectors:
            return True

        sizes = [len(v) for v in self.flattened_vectors]
        uniform = len(set(sizes)) == 1

        if not uniform:
            print(f"⚠ WARNING: Vectors have non-uniform sizes: {sorted(set(sizes))}")
            for i, (filename, size) in enumerate(zip(self.filenames, sizes)):
                print(f"  [{i}] {Path(filename).name}: {size} elements")
        else:
            print(f"✓ All {len(self.flattened_vectors)} vectors have uniform size: {sizes[0]}")

        return uniform

    def __len__(self):
        return len(self.networks)

    def __repr__(self):
        vector_length = len(self.flattened_vectors[0]) if self.flattened_vectors else 0
        return f"NetworkSeries(n_networks={len(self.networks)}, vector_length={vector_length})"


def load_models_from_folder(folder_path: str, pattern: str = "*.bnet") -> NetworkSeries:
    """Load all .bnet models from a folder and convert to flattened vectors."""
    network_series = NetworkSeries()

    bnet_files = glob.glob(os.path.join(folder_path, "**", pattern), recursive=True)

    if not bnet_files:
        print(f"Warning: No {pattern} files found in {folder_path}")
        return network_series

    print(f"Found {len(bnet_files)} model(s) in {folder_path}")

    for bnet_file in sorted(bnet_files):
        try:
            edges_df, nodes = parse_bnet_file(bnet_file)
            feature_matrix = create_feature_matrix(edges_df, nodes)
            flattened_vector = feature_matrix.values.flatten()

            network_series.add_network(
                filename=bnet_file,
                edges_df=edges_df,
                nodes=nodes,
                feature_matrix=feature_matrix,
                flattened_vector=flattened_vector,
            )

            print(f"  ✓ Loaded: {Path(bnet_file).name} ({len(nodes)} nodes, {len(flattened_vector)} edges)")

        except Exception as e:
            print(f"  ✗ Error loading {bnet_file}: {e}")

    return network_series
