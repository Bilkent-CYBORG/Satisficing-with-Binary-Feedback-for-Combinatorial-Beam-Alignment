"""Hierarchical codebook for beam management."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import deepmimo as dm
import numpy as np


@dataclass
class BeamNode:
    """A node in the hierarchical codebook tree.

    Attributes
    ----------
    level : int
        Hierarchical level (0 = coarsest, H-1 = finest).
    idx : int
        Index within the level.
    beam_vector : np.ndarray
        Complex beamforming vector of shape (N,).
    coverage_start : float
        Start angle of coverage region in degrees.
    coverage_end : float
        End angle of coverage region in degrees.
    zoom_in_idxs : list of int
        List of children indices at level+1.
    zoom_out_idx : int or None
        Parent index at level-1 (None for level 0).
    """

    level: int
    idx: int
    beam_vector: np.ndarray
    coverage_start: float
    coverage_end: float
    zoom_in_idxs: List[int] = field(default_factory=list)
    zoom_out_idx: Optional[int] = None

    @property
    def beamwidth(self) -> float:
        """Coverage beamwidth in degrees."""
        return abs(self.coverage_end - self.coverage_start)

    @property
    def center_angle(self) -> float:
        """Center angle of coverage region in degrees."""
        return (self.coverage_start + self.coverage_end) / 2


class HierarchicalCodebook:
    """Hierarchical codebook with wider beams at coarser levels.

    Uses sum + normalize approach to create wider beams from narrow beams.
    Binary tree structure (g=2) where each parent has 2 children.

    Attributes
    ----------
    K : int
        Number of beams at finest level.
    N : int
        Number of antenna elements.
    H : int
        Number of hierarchical levels.
    g : int
        Branching factor (default 2).
    codebooks : dict
        Dict mapping level to codebook array (num_beams, N).
    nodes : dict
        Dict mapping (level, idx) to BeamNode.
    beam_angles : np.ndarray
        Array of beam angles at finest level.
    """

    def __init__(
        self,
        K: int,
        N: int,
        antenna_shape: Tuple[int, ...],
        num_levels: Optional[int] = None,
        g: int = 2,
    ):
        """Initialize hierarchical codebook.

        Parameters
        ----------
        K : int
            Number of beams at finest level.
        N : int
            Number of antenna elements.
        antenna_shape : tuple of int
            Antenna array shape for dm.steering_vec.
        num_levels : int, optional
            Number of levels. Default is log2(K).
        g : int, optional
            Branching factor. Default is 2 for binary tree.
        """
        self.K = K
        self.N = N
        self.antenna_shape = antenna_shape
        self.g = g

        # Calculate number of levels
        if num_levels is None:
            self.H = int(np.log2(K))
        else:
            self.H = num_levels

        # Storage
        self.codebooks: Dict[int, np.ndarray] = {}
        self.nodes: Dict[Tuple[int, int], BeamNode] = {}
        self.beam_angles: np.ndarray = None

        # Build hierarchy
        self._build_hierarchy()

    def _build_hierarchy(self):
        """Build the hierarchical codebook structure."""
        # Step 1: Generate finest level codebook (standard DFT)
        finest_level = self.H - 1
        finest_codebook = np.zeros((self.K, self.N), dtype=np.complex128)
        self.beam_angles = np.zeros(self.K)

        for k in range(self.K):
            u = -1.0 + (2.0 * k) / self.K
            phi_deg = np.degrees(np.arcsin(np.clip(u, -1.0, 1.0)))
            self.beam_angles[k] = phi_deg

            dm_vec = dm.steering_vec(self.antenna_shape, phi=float(phi_deg)).squeeze()
            dm_vec = dm_vec / (np.linalg.norm(dm_vec) + 1e-12)
            finest_codebook[k] = dm_vec

        self.codebooks[finest_level] = finest_codebook

        # Step 2: Create finest level nodes
        beamwidth = 180.0 / self.K
        for k in range(self.K):
            node = BeamNode(
                level=finest_level,
                idx=k,
                beam_vector=finest_codebook[k],
                coverage_start=self.beam_angles[k] - beamwidth / 2,
                coverage_end=self.beam_angles[k] + beamwidth / 2,
            )
            self.nodes[(finest_level, k)] = node

        # Step 3: Build coarser levels by summing children
        for level in range(finest_level - 1, -1, -1):
            num_beams = self._beams_at_level(level)
            level_codebook = np.zeros((num_beams, self.N), dtype=np.complex128)

            for i in range(num_beams):
                # Get children indices at level+1
                child_start = i * self.g
                child_end = min(child_start + self.g, self._beams_at_level(level + 1))
                child_indices = list(range(child_start, child_end))

                # Sum children beamforming vectors
                wide_beam = np.zeros(self.N, dtype=np.complex128)
                for child_idx in child_indices:
                    child_node = self.nodes[(level + 1, child_idx)]
                    wide_beam += child_node.beam_vector
                    child_node.zoom_out_idx = i

                # Normalize
                wide_beam = wide_beam / (np.linalg.norm(wide_beam) + 1e-12)
                level_codebook[i] = wide_beam

                # Create node with coverage from first to last child
                first_child = self.nodes[(level + 1, child_indices[0])]
                last_child = self.nodes[(level + 1, child_indices[-1])]

                node = BeamNode(
                    level=level,
                    idx=i,
                    beam_vector=wide_beam,
                    coverage_start=first_child.coverage_start,
                    coverage_end=last_child.coverage_end,
                    zoom_in_idxs=child_indices,
                )
                self.nodes[(level, i)] = node

            self.codebooks[level] = level_codebook

    def _beams_at_level(self, level: int) -> int:
        """Number of beams at a given level."""
        finest_level = self.H - 1
        return self.K // (self.g ** (finest_level - level))

    def num_beams_at_level(self, level: int) -> int:
        """Public method: number of beams at a given level."""
        return self._beams_at_level(level)

    def get_codebook(self, level: int) -> np.ndarray:
        """Get codebook at a specific level.

        Parameters
        ----------
        level : int
            Hierarchical level.

        Returns
        -------
        np.ndarray
            Codebook array of shape (num_beams_at_level, N).
        """
        return self.codebooks[level]

    def get_node(self, level: int, idx: int) -> BeamNode:
        """Get a beam node.

        Parameters
        ----------
        level : int
            Hierarchical level.
        idx : int
            Beam index at that level.

        Returns
        -------
        BeamNode
            BeamNode instance.
        """
        return self.nodes[(level, idx)]

    def get_children(self, level: int, idx: int) -> List[BeamNode]:
        """Get children nodes of a beam.

        Parameters
        ----------
        level : int
            Current level.
        idx : int
            Beam index.

        Returns
        -------
        list of BeamNode
            List of child BeamNodes (empty if at finest level).
        """
        node = self.nodes[(level, idx)]
        return [self.nodes[(level + 1, child_idx)] for child_idx in node.zoom_in_idxs]

    def get_parent(self, level: int, idx: int) -> Optional[BeamNode]:
        """Get parent node of a beam.

        Parameters
        ----------
        level : int
            Current level.
        idx : int
            Beam index.

        Returns
        -------
        BeamNode or None
            Parent BeamNode or None if at coarsest level.
        """
        node = self.nodes[(level, idx)]
        if node.zoom_out_idx is None:
            return None
        return self.nodes[(level - 1, node.zoom_out_idx)]

    def get_siblings(self, level: int, idx: int) -> List[BeamNode]:
        """Get sibling nodes (same parent).

        Parameters
        ----------
        level : int
            Current level.
        idx : int
            Beam index.

        Returns
        -------
        list of BeamNode
            List of sibling BeamNodes including self.
        """
        parent = self.get_parent(level, idx)
        if parent is None:
            # At level 0, all beams are siblings
            return [self.nodes[(level, i)] for i in range(self._beams_at_level(level))]
        return self.get_children(parent.level, parent.idx)

    def print_hierarchy(self):
        """Print the hierarchy structure."""
        print(f"Hierarchical Codebook: K={self.K}, N={self.N}, H={self.H}, g={self.g}")
        print("-" * 60)
        for level in range(self.H):
            num_beams = self._beams_at_level(level)
            print(f"Level {level}: {num_beams} beams")
            for i in range(num_beams):
                node = self.nodes[(level, i)]
                children = node.zoom_in_idxs if node.zoom_in_idxs else "leaf"
                parent = node.zoom_out_idx if node.zoom_out_idx is not None else "root"
                print(
                    f"  [{i}] angle={node.center_angle:.1f} deg, "
                    f"width={node.beamwidth:.1f} deg, "
                    f"parent={parent}, children={children}"
                )
        print("-" * 60)

    @property
    def finest_codebook(self) -> np.ndarray:
        """Get the finest level codebook (same as standard DFT codebook)."""
        return self.codebooks[self.H - 1]

    @property
    def coarsest_codebook(self) -> np.ndarray:
        """Get the coarsest level codebook."""
        return self.codebooks[0]

    @property
    def total_beams(self) -> int:
        """Total number of beams across all levels (2K - 2 for binary tree)."""
        return 2 * self.K - 2

    def flat_idx_to_level_beam(self, flat_idx: int) -> Tuple[int, int]:
        """Convert flat index to (level, beam_idx).

        Parameters
        ----------
        flat_idx : int
            Flat index in range [0, total_beams).

        Returns
        -------
        tuple of (int, int)
            Tuple of (level, beam_idx).
        """
        offset = 0
        for level in range(self.H):
            num_beams = self.num_beams_at_level(level)
            if flat_idx < offset + num_beams:
                return level, flat_idx - offset
            offset += num_beams
        return self.H - 1, flat_idx - offset

    def level_beam_to_flat_idx(self, level: int, beam_idx: int) -> int:
        """Convert (level, beam_idx) to flat index.

        Parameters
        ----------
        level : int
            Hierarchical level.
        beam_idx : int
            Beam index within level.

        Returns
        -------
        int
            Flat index in range [0, total_beams).
        """
        offset = 0
        for lv in range(level):
            offset += self.num_beams_at_level(lv)
        return offset + beam_idx
