import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D


def _validate_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    return mol


def _extract_atom_weights(alpha_padded: Sequence[float], num_atoms: int) -> List[float]:
    if len(alpha_padded) < num_atoms:
        raise ValueError(
            f"alpha_padded length ({len(alpha_padded)}) is smaller than num_atoms ({num_atoms})."
        )
    return list(alpha_padded[:num_atoms])


def draw_single_family_heatmap(
    smiles: str,
    atom_weights: Sequence[float],
    family_name: str,
    out_path: str,
    size: Tuple[int, int] = (700, 500),
    contour_lines: int = 10,
) -> str:
    """
    Draw a single RDKit atom-level attention heatmap for one descriptor family.
    """
    mol = _validate_mol(smiles)
    n_atoms = mol.GetNumAtoms()
    if len(atom_weights) != n_atoms:
        raise ValueError(
            f"len(atom_weights)={len(atom_weights)} must equal number of atoms={n_atoms}."
        )

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        list(atom_weights),
        draw2d=drawer,
        contourLines=contour_lines,
    )
    drawer.FinishDrawing()

    with open(out_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    return out_path


def draw_family_beta_barplot(
    beta_dict: Dict[str, float],
    out_path: str,
    title: str = "Family Importance (beta)",
    figsize: Tuple[int, int] = (8, 4),
) -> str:
    """
    Draw a bar plot for family-level aggregation weights beta.
    """
    families = list(beta_dict.keys())
    values = [float(beta_dict[f]) for f in families]

    plt.figure(figsize=figsize)
    plt.bar(families, values)
    plt.title(title)
    plt.ylabel("Attention weight")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def draw_all_family_heatmaps_and_beta(
    smiles: str,
    alpha_dict: Dict[str, Sequence[float]],
    beta_dict: Dict[str, float],
    out_dir: str = ".",
    prefix: str = "sample",
    contour_lines: int = 10,
    heatmap_size: Tuple[int, int] = (700, 500),
    draw_combined_beta: bool = True,
) -> Dict[str, str]:
    """
    Draw one heatmap per family and one beta barplot.

    Args:
        smiles:
            Molecule SMILES.
        alpha_dict:
            Dict[str, Sequence[float]]
            Example:
                {
                    "constitutional": [padded alpha...],
                    "topological": [padded alpha...],
                    ...
                }
            Each family alpha can be padded [N_max]. It will be truncated to num_atoms.
        beta_dict:
            Dict[str, float] of family importance weights.
        out_dir:
            Output directory.
        prefix:
            Prefix for filenames.
        contour_lines:
            Number of contour lines in RDKit similarity map.
        heatmap_size:
            (width, height) for RDKit heatmaps.
        draw_combined_beta:
            Whether to create the family-level beta bar plot.

    Returns:
        Dict[str, str] mapping artifact names to file paths.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    mol = _validate_mol(smiles)
    num_atoms = mol.GetNumAtoms()

    saved = {}

    # Draw per-family atom heatmaps
    for fam, alpha_padded in alpha_dict.items():
        atom_weights = _extract_atom_weights(alpha_padded, num_atoms)
        out_path = os.path.join(out_dir, f"{prefix}_{fam}_heatmap.png")
        draw_single_family_heatmap(
            smiles=smiles,
            atom_weights=atom_weights,
            family_name=fam,
            out_path=out_path,
            size=heatmap_size,
            contour_lines=contour_lines,
        )
        saved[f"{fam}_heatmap"] = out_path

    # Draw beta barplot
    if draw_combined_beta:
        beta_path = os.path.join(out_dir, f"{prefix}_beta_barplot.png")
        draw_family_beta_barplot(
            beta_dict=beta_dict,
            out_path=beta_path,
            title=f"{prefix}: Family Importance (beta)",
        )
        saved["beta_barplot"] = beta_path

    return saved


def convert_model_outputs_for_one_sample(
    smiles: str,
    attn_dict,
    beta,
    family_order: Optional[Sequence[str]] = None,
    sample_idx: int = 0,
):
    """
    Helper for converting model outputs into visualization-ready dictionaries.

    Expected inputs:
        attn_dict[fam]: torch.Tensor [B, N_max]
        beta: torch.Tensor [B, K]
        family_order: list like
            ['constitutional', 'topological', 'physicochemical', 'electronic', 'fragment']

    Returns:
        alpha_dict: dict[fam] -> padded alpha list
        beta_dict:  dict[fam] -> scalar beta
    """
    try:
        import torch  # noqa: F401
    except Exception:
        pass

#     if family_order is None:
#         family_order = list(attn_dict.keys())

#     alpha_dict = {
#         fam: attn_dict[fam][sample_idx].detach().cpu().tolist()
#         for fam in family_order
#     }

#     beta_row = beta[sample_idx].detach().cpu().tolist()
#     beta_dict = {fam: float(beta_row[i]) for i, fam in enumerate(family_order)}

#     return alpha_dict, beta_dict


# # if __name__ == "__main__":
# #     import os

# #     # Example molecule
# #     smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
# #     mol = Chem.MolFromSmiles(smiles)
# #     n = mol.GetNumAtoms()

# #     # Example family attention weights (pretend these came from your model's attn_dict)
# #     alpha_dict = {
# #         "constitutional": [0.04, 0.06, 0.05, 0.04, 0.10, 0.12, 0.10, 0.08, 0.07, 0.08, 0.11, 0.08, 0.07],
# #         "topological":    [0.05, 0.05, 0.04, 0.03, 0.16, 0.17, 0.15, 0.12, 0.07, 0.06, 0.04, 0.03, 0.03],
# #         "physicochemical":[0.03, 0.06, 0.06, 0.04, 0.18, 0.20, 0.14, 0.08, 0.05, 0.06, 0.22, 0.16, 0.12],
# #         "electronic":     [0.02, 0.08, 0.09, 0.05, 0.12, 0.11, 0.10, 0.08, 0.05, 0.08, 0.28, 0.22, 0.20],
# #         "fragment":       [0.03, 0.07, 0.08, 0.04, 0.09, 0.10, 0.09, 0.07, 0.05, 0.06, 0.30, 0.26, 0.18],
# #     }

# #     # Normalize each family for a cleaner mock example
# #     for k, v in alpha_dict.items():
# #         s = sum(v)
# #         alpha_dict[k] = [x / s for x in v]

# #     beta_dict = {
# #         "constitutional": 0.11,
# #         "topological": 0.18,
# #         "physicochemical": 0.24,
# #         "electronic": 0.29,
# #         "fragment": 0.18,
# #     }

# #     out_dir = "."
# #     files = draw_all_family_heatmaps_and_beta(
# #         smiles=smiles,
# #         alpha_dict=alpha_dict,
# #         beta_dict=beta_dict,
# #         out_dir=out_dir,
# #         prefix="aspirin",
# #     )

# #     print("Saved files:")
# #     for k, v in files.items():
# #         print(f"{k}: {os.path.abspath(v)}")





# import math
# import os
# from typing import Dict, List, Optional, Sequence, Tuple

# import matplotlib.pyplot as plt
# from rdkit import Chem
# from rdkit.Chem import rdDepictor
# from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D


# def _validate_mol(smiles: str) -> Chem.Mol:
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smiles}")
#     rdDepictor.Compute2DCoords(mol)
#     return mol


# def _extract_atom_weights(alpha_padded: Sequence[float], num_atoms: int) -> List[float]:
#     if len(alpha_padded) < num_atoms:
#         raise ValueError(
#             f"alpha_padded length ({len(alpha_padded)}) is smaller than num_atoms ({num_atoms})."
#         )
#     return list(alpha_padded[:num_atoms])


# def _minmax_normalize(weights: Sequence[float], eps: float = 1e-12) -> List[float]:
#     w_min = min(weights)
#     w_max = max(weights)
#     if abs(w_max - w_min) < eps:
#         return [0.5 for _ in weights]
#     return [(w - w_min) / (w_max - w_min) for w in weights]


# def _residual_from_uniform(weights: Sequence[float]) -> List[float]:
#     """
#     alpha_i - 1/N
#     Positive: above-uniform attention
#     Negative: below-uniform attention
#     """
#     n = len(weights)
#     base = 1.0 / n
#     return [w - base for w in weights]


# def _signed_contrast_transform(
#     weights: Sequence[float],
#     gamma: float = 0.35,
#     eps: float = 1e-12,
# ) -> List[float]:
#     """
#     Make contrast much stronger while preserving sign.
#     Smaller gamma (<1) exaggerates differences.
#     """
#     max_abs = max(abs(w) for w in weights)
#     if max_abs < eps:
#         return [0.0 for _ in weights]

#     out = []
#     for w in weights:
#         s = 1.0 if w >= 0 else -1.0
#         mag = abs(w) / max_abs
#         mag = mag ** gamma
#         out.append(s * mag)
#     return out


# def _prepare_weights_for_visualization(
#     atom_weights: Sequence[float],
#     mode: str = "residual_contrast",
#     gamma: float = 0.35,
# ) -> List[float]:
#     """
#     mode:
#         - raw: raw alpha
#         - minmax: family-wise min-max normalization to [0,1]
#         - residual: alpha - 1/N
#         - residual_contrast: residual + signed power transform (recommended)
#     """
#     weights = list(atom_weights)

#     if mode == "raw":
#         return weights

#     if mode == "minmax":
#         return _minmax_normalize(weights)

#     if mode == "residual":
#         return _residual_from_uniform(weights)

#     if mode == "residual_contrast":
#         residual = _residual_from_uniform(weights)
#         return _signed_contrast_transform(residual, gamma=gamma)

#     raise ValueError(f"Unknown mode: {mode}")


# def draw_single_family_heatmap(
#     smiles: str,
#     atom_weights: Sequence[float],
#     family_name: str,
#     out_path: str,
#     size: Tuple[int, int] = (700, 500),
#     contour_lines: int = 0,
#     weight_mode: str = "residual_contrast",
#     gamma: float = 0.35,
# ) -> str:
#     """
#     Draw a high-contrast RDKit atom-level attention heatmap.

#     Recommended:
#         weight_mode="residual_contrast"
#     """
#     mol = _validate_mol(smiles)
#     n_atoms = mol.GetNumAtoms()
#     if len(atom_weights) != n_atoms:
#         raise ValueError(
#             f"len(atom_weights)={len(atom_weights)} must equal number of atoms={n_atoms}."
#         )

#     viz_weights = _prepare_weights_for_visualization(
#         atom_weights,
#         mode=weight_mode,
#         gamma=gamma,
#     )

#     drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

#     # Stronger molecule line visibility
#     opts = drawer.drawOptions()
#     opts.bondLineWidth = 2
#     opts.legendFontSize = 18

#     SimilarityMaps.GetSimilarityMapFromWeights(
#         mol,
#         list(viz_weights),
#         draw2d=drawer,
#         contourLines=contour_lines,
#     )

#     drawer.FinishDrawing()
#     with open(out_path, "wb") as f:
#         f.write(drawer.GetDrawingText())

#     return out_path


# def draw_family_beta_barplot(
#     beta_dict: Dict[str, float],
#     out_path: str,
#     title: str = "Family Importance (beta)",
#     figsize: Tuple[int, int] = (8, 4),
# ) -> str:
#     families = list(beta_dict.keys())
#     values = [float(beta_dict[f]) for f in families]

#     plt.figure(figsize=figsize)
#     plt.bar(families, values)
#     plt.title(title)
#     plt.ylabel("Attention weight")
#     plt.xticks(rotation=20, ha="right")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200, bbox_inches="tight")
#     plt.close()
#     return out_path


# def draw_family_alpha_barplot(
#     alpha_dict: Dict[str, Sequence[float]],
#     out_dir: str,
#     prefix: str,
# ) -> Dict[str, str]:
#     """
#     Save simple atom-index barplots per family.
#     This is often more informative than molecule heatmaps when attention differences are small.
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     saved = {}

#     for fam, weights in alpha_dict.items():
#         idx = list(range(len(weights)))
#         plt.figure(figsize=(8, 3))
#         plt.bar(idx, weights)
#         plt.title(f"{prefix}: {fam} atom attention")
#         plt.xlabel("Atom index")
#         plt.ylabel("alpha")
#         plt.tight_layout()

#         out_path = os.path.join(out_dir, f"{prefix}_{fam}_alpha_barplot.png")
#         plt.savefig(out_path, dpi=200, bbox_inches="tight")
#         plt.close()

#         saved[f"{fam}_alpha_barplot"] = out_path

#     return saved


# def make_beta_weighted_atom_scores(
#     alpha_dict: Dict[str, Sequence[float]],
#     beta_dict: Dict[str, float],
# ) -> List[float]:
#     """
#     gamma_i = sum_k beta_k * alpha_{k,i}
#     """
#     family_order = list(alpha_dict.keys())
#     num_atoms = len(next(iter(alpha_dict.values())))
#     gamma = [0.0] * num_atoms

#     for fam in family_order:
#         beta_k = float(beta_dict[fam])
#         alpha_k = alpha_dict[fam]
#         for i in range(num_atoms):
#             gamma[i] += beta_k * alpha_k[i]

#     return gamma


# def draw_all_family_heatmaps_and_beta(
#     smiles: str,
#     alpha_dict: Dict[str, Sequence[float]],
#     beta_dict: Dict[str, float],
#     out_dir: str = ".",
#     prefix: str = "sample",
#     contour_lines: int = 0,
#     heatmap_size: Tuple[int, int] = (700, 500),
#     draw_combined_beta: bool = True,
#     draw_alpha_barplots: bool = True,
#     draw_global_heatmap: bool = True,
#     weight_mode: str = "residual_contrast",
#     gamma: float = 0.35,
# ) -> Dict[str, str]:
#     """
#     High-contrast visualization bundle.

#     Recommended:
#         weight_mode="residual_contrast"
#         gamma=0.35
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     mol = _validate_mol(smiles)
#     num_atoms = mol.GetNumAtoms()

#     saved = {}

#     # Per-family atom heatmaps
#     for fam, alpha_padded in alpha_dict.items():
#         atom_weights = _extract_atom_weights(alpha_padded, num_atoms)
#         out_path = os.path.join(out_dir, f"{prefix}_{fam}_heatmap.png")

#         draw_single_family_heatmap(
#             smiles=smiles,
#             atom_weights=atom_weights,
#             family_name=fam,
#             out_path=out_path,
#             size=heatmap_size,
#             contour_lines=contour_lines,
#             weight_mode=weight_mode,
#             gamma=gamma,
#         )
#         saved[f"{fam}_heatmap"] = out_path

#     # Global atom heatmap using beta-weighted family attention
#     if draw_global_heatmap:
#         trimmed_alpha_dict = {
#             fam: _extract_atom_weights(alpha_dict[fam], num_atoms)
#             for fam in alpha_dict.keys()
#         }
#         global_weights = make_beta_weighted_atom_scores(trimmed_alpha_dict, beta_dict)

#         global_path = os.path.join(out_dir, f"{prefix}_global_heatmap.png")
#         draw_single_family_heatmap(
#             smiles=smiles,
#             atom_weights=global_weights,
#             family_name="global",
#             out_path=global_path,
#             size=heatmap_size,
#             contour_lines=contour_lines,
#             weight_mode=weight_mode,
#             gamma=gamma,
#         )
#         saved["global_heatmap"] = global_path

#     # Beta barplot
#     if draw_combined_beta:
#         beta_path = os.path.join(out_dir, f"{prefix}_beta_barplot.png")
#         draw_family_beta_barplot(
#             beta_dict=beta_dict,
#             out_path=beta_path,
#             title=f"{prefix}: Family Importance (beta)",
#         )
#         saved["beta_barplot"] = beta_path

#     # Per-family alpha barplots
#     if draw_alpha_barplots:
#         trimmed_alpha_dict = {
#             fam: _extract_atom_weights(alpha_dict[fam], num_atoms)
#             for fam in alpha_dict.keys()
#         }
#         saved.update(draw_family_alpha_barplot(trimmed_alpha_dict, out_dir, prefix))

#     return saved


# def convert_model_outputs_for_one_sample(
#     smiles: str,
#     attn_dict,
#     beta,
#     family_order: Optional[Sequence[str]] = None,
#     sample_idx: int = 0,
# ):
#     """
#     Expected:
#         attn_dict[fam]: torch.Tensor [B, N_max]
#         beta: torch.Tensor [B, K]
#     """
#     try:
#         import torch  # noqa: F401
#     except Exception:
#         pass

#     if family_order is None:
#         family_order = list(attn_dict.keys())

#     alpha_dict = {
#         fam: attn_dict[fam][sample_idx].detach().cpu().tolist()
#         for fam in family_order
#     }

#     beta_row = beta[sample_idx].detach().cpu().tolist()
#     beta_dict = {fam: float(beta_row[i]) for i, fam in enumerate(family_order)}

#     return alpha_dict, beta_dict