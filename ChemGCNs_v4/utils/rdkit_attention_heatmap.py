from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D

def draw_attention_heatmap(
    smiles: str,
    atom_weights,
    out_path: str = "attention_heatmap.png",
    size=(700, 500),
    contour_lines: int = 10,
):
    """
    Draw an RDKit atom-level heatmap from attention weights.

    Args:
        smiles: SMILES string of a molecule.
        atom_weights: list/tuple of length == num_atoms.
                      Example: one family's alpha for a single molecule.
        out_path: output PNG path.
        size: (width, height)
        contour_lines: number of contour lines in the heatmap.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    n_atoms = mol.GetNumAtoms()
    if len(atom_weights) != n_atoms:
        raise ValueError(
            f"len(atom_weights)={len(atom_weights)} must equal num_atoms={n_atoms}"
        )

    rdDepictor.Compute2DCoords(mol)
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


def extract_single_molecule_alpha(alpha_padded, num_atoms: int):
    """
    Convert padded alpha [N_max] to real atom-level weights [num_atoms].
    """
    if len(alpha_padded) < num_atoms:
        raise ValueError("num_atoms exceeds padded attention length.")
    return list(alpha_padded[:num_atoms])


if __name__ == "__main__":
    # Example: aspirin
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    weights = [0.04, 0.08, 0.06, 0.03, 0.22, 0.28, 0.17, 0.10, 0.05, 0.07, 0.32, 0.24, 0.18]
    draw_attention_heatmap(smiles, weights, "sample_attention_heatmap.png")
    print("Saved sample_attention_heatmap.png")
