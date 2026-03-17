import numpy as np
import torch

import dgl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold_id_map(smiles_list):
    """
    SMILES 리스트로부터 Murcko scaffold 기반 정수 ID 매핑 생성.

    Returns:
        scaffold_ids  : list[int]  각 분자의 scaffold 정수 ID (-1: 파싱 실패)
        n_scaffolds   : int        고유 scaffold 수
    """
    scaffold_to_id: dict[str, int] = {}
    scaffold_ids:   list[int]      = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold_ids.append(-1)
            continue
        scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        if scaffold_smi not in scaffold_to_id:
            scaffold_to_id[scaffold_smi] = len(scaffold_to_id)
        scaffold_ids.append(scaffold_to_id[scaffold_smi])

    return scaffold_ids, len(scaffold_to_id)

# Freesolv
def descriptor_selection_freesolv(samples):
    self_feats = np.empty((len(samples), 50), dtype=np.float32)
    smiles_list=[]

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        # 1
        self_feats[i, 0] = mol_graph.NHOHCount
        self_feats[i, 1] = mol_graph.SlogP_VSA2
        self_feats[i, 2] = mol_graph.SlogP_VSA10
        self_feats[i, 3] = mol_graph.NumAromaticRings
        self_feats[i, 4] = mol_graph.MaxEStateIndex
        # 6
        self_feats[i, 5] = mol_graph.PEOE_VSA14
        self_feats[i, 6] = mol_graph.fr_Ar_NH
        self_feats[i, 7] = mol_graph.SMR_VSA3
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.SlogP_VSA5
        # 11
        self_feats[i, 10] = mol_graph.VSA_EState8
        self_feats[i, 11] = mol_graph.MaxAbsEStateIndex
        self_feats[i, 12] = mol_graph.PEOE_VSA2
        self_feats[i, 13] = mol_graph.fr_Nhpyrrole
        self_feats[i, 14] = mol_graph.fr_amide
        # 16
        self_feats[i, 15] = mol_graph.SlogP_VSA3
        self_feats[i, 16] = mol_graph.BCUT2D_MRHI
        self_feats[i, 17] = mol_graph.fr_nitrile
        self_feats[i, 18] = mol_graph.MolLogP
        self_feats[i, 19] = mol_graph.PEOE_VSA10
        # 21
        self_feats[i, 20] = mol_graph.MinPartialCharge
        self_feats[i, 21] = mol_graph.fr_Al_OH
        self_feats[i, 22] = mol_graph.fr_sulfone
        self_feats[i, 23] = mol_graph.fr_Al_COO
        self_feats[i, 24] = mol_graph.fr_nitro_arom_nonortho
        # 26
        self_feats[i, 25] = mol_graph.fr_imidazole
        self_feats[i, 26] = mol_graph.fr_ketone_Topliss
        self_feats[i, 27] = mol_graph.PEOE_VSA7
        self_feats[i, 28] = mol_graph.fr_alkyl_halide
        self_feats[i, 29] = mol_graph.NumSaturatedHeterocycles
        # 31
        self_feats[i, 30] = mol_graph.fr_methoxy
        self_feats[i, 31] = mol_graph.fr_phos_acid
        self_feats[i, 32] = mol_graph.fr_pyridine
        self_feats[i, 33] = mol_graph.MinAbsEStateIndex
        self_feats[i, 34] = mol_graph.fr_para_hydroxylation
        # 36
        self_feats[i, 35] = mol_graph.fr_phos_ester
        self_feats[i, 36] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 37] = mol_graph.PEOE_VSA8
        self_feats[i, 38] = mol_graph.fr_Ndealkylation2
        self_feats[i, 39] = mol_graph.PEOE_VSA5
        # 41
        self_feats[i, 40] = mol_graph.fr_aryl_methyl
        self_feats[i, 41] = mol_graph.NumHDonors
        self_feats[i, 42] = mol_graph.fr_imide
        self_feats[i, 43] = mol_graph.fr_priamide
        self_feats[i, 44] = mol_graph.RingCount
        # 46
        self_feats[i, 45] = mol_graph.SlogP_VSA8
        self_feats[i, 46] = mol_graph.VSA_EState4
        self_feats[i, 47] = mol_graph.SMR_VSA5
        self_feats[i, 48] = mol_graph.FpDensityMorgan3
        self_feats[i, 49] = mol_graph.FractionCSP3

        smiles_list.append(mol_graph.smiles)
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    
    scaffold_ids, le= get_scaffold_id_map(smiles_list)
    
    return (
        batched_graph, 
        torch.tensor(self_feats).to(device), 
        torch.tensor(labels, dtype=torch.float32).to(device),
        torch.tensor(scaffold_ids, dtype=torch.long).to(device),
        smiles_list)


# ESOL
def descriptor_selection_esol(samples):
    self_feats = np.empty((len(samples), 63), dtype=np.float32)
    smiles_list=[]

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.MaxAbsPartialCharge
        self_feats[i, 2] = mol_graph.MaxEStateIndex
        self_feats[i, 3] = mol_graph.SMR_VSA10
        self_feats[i, 4] = mol_graph.Kappa2
        # 6
        self_feats[i, 5] = mol_graph.BCUT2D_MWLOW
        self_feats[i, 6] = mol_graph.PEOE_VSA13
        self_feats[i, 7] = mol_graph.MinAbsPartialCharge
        self_feats[i, 8] = mol_graph.BCUT2D_CHGHI
        self_feats[i, 9] = mol_graph.PEOE_VSA6
        # 11
        self_feats[i, 10] = mol_graph.SlogP_VSA1
        self_feats[i, 11] = mol_graph.fr_nitro
        self_feats[i, 12] = mol_graph.BalabanJ
        self_feats[i, 13] = mol_graph.SMR_VSA9
        self_feats[i, 14] = mol_graph.fr_alkyl_halide
        # 16
        self_feats[i, 15] = mol_graph.fr_hdrzine
        self_feats[i, 16] = mol_graph.PEOE_VSA8
        self_feats[i, 17] = mol_graph.fr_Ar_NH
        self_feats[i, 18] = mol_graph.fr_imidazole
        self_feats[i, 19] = mol_graph.fr_Nhpyrrole
        # 21
        self_feats[i, 20] = mol_graph.EState_VSA5
        self_feats[i, 21] = mol_graph.PEOE_VSA4
        self_feats[i, 22] = mol_graph.fr_ester
        self_feats[i, 23] = mol_graph.PEOE_VSA2
        self_feats[i, 24] = mol_graph.NumAromaticCarbocycles
        # 26
        self_feats[i, 25] = mol_graph.BCUT2D_LOGPHI
        self_feats[i, 26] = mol_graph.EState_VSA11
        self_feats[i, 27] = mol_graph.fr_furan
        self_feats[i, 28] = mol_graph.EState_VSA2
        self_feats[i, 29] = mol_graph.fr_benzene
        # 31
        self_feats[i, 30] = mol_graph.fr_sulfide
        self_feats[i, 31] = mol_graph.fr_aryl_methyl
        self_feats[i, 32] = mol_graph.SlogP_VSA10
        self_feats[i, 33] = mol_graph.HeavyAtomMolWt
        self_feats[i, 34] = mol_graph.fr_nitro_arom_nonortho
        # 36
        self_feats[i, 35] = mol_graph.FpDensityMorgan2
        self_feats[i, 36] = mol_graph.EState_VSA8
        self_feats[i, 37] = mol_graph.fr_bicyclic
        self_feats[i, 38] = mol_graph.fr_aniline
        self_feats[i, 39] = mol_graph.fr_allylic_oxid
        # 41
        self_feats[i, 40] = mol_graph.fr_C_S
        self_feats[i, 41] = mol_graph.SlogP_VSA7
        self_feats[i, 42] = mol_graph.SlogP_VSA4
        self_feats[i, 43] = mol_graph.fr_para_hydroxylation
        self_feats[i, 44] = mol_graph.PEOE_VSA7
        # 46
        self_feats[i, 45] = mol_graph.fr_Al_OH_noTert
        self_feats[i, 46] = mol_graph.fr_pyridine
        self_feats[i, 47] = mol_graph.fr_phos_acid
        self_feats[i, 48] = mol_graph.fr_phos_ester
        self_feats[i, 49] = mol_graph.NumAromaticHeterocycles
        # 51
        self_feats[i, 50] = mol_graph.EState_VSA7
        self_feats[i, 51] = mol_graph.PEOE_VSA12
        self_feats[i, 52] = mol_graph.Ipc
        self_feats[i, 53] = mol_graph.FpDensityMorgan1
        self_feats[i, 54] = mol_graph.PEOE_VSA14
        # 56
        self_feats[i, 55] = mol_graph.fr_guanido
        self_feats[i, 56] = mol_graph.fr_benzodiazepine
        self_feats[i, 57] = mol_graph.fr_thiophene
        self_feats[i, 58] = mol_graph.fr_Ndealkylation1
        self_feats[i, 59] = mol_graph.fr_aldehyde
        # 61
        self_feats[i, 60] = mol_graph.fr_term_acetylene
        self_feats[i, 61] = mol_graph.SMR_VSA2
        self_feats[i, 62] = mol_graph.fr_lactone

        smiles_list.append(mol_graph.smiles)
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    scaffold_ids, le= get_scaffold_id_map(smiles_list)
    
    return (
        batched_graph, 
        torch.tensor(self_feats).to(device), 
        torch.tensor(labels, dtype=torch.float32).to(device),
        torch.tensor(scaffold_ids, dtype=torch.long).to(device),
        smiles_list)

# lipo
def descriptor_selection_lipo(samples):
    self_feats = np.empty((len(samples), 25), dtype=np.float32)
    smiles_list=[]

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        # 1
        self_feats[i, 0] = mol_graph.MolLogP
        self_feats[i, 1] = mol_graph.fr_COO
        self_feats[i, 2] = mol_graph.Ipc
        self_feats[i, 3] = mol_graph.fr_sulfonamd
        self_feats[i, 4] = mol_graph.PEOE_VSA7
        # 6
        self_feats[i, 5] = mol_graph.PEOE_VSA13
        self_feats[i, 6] = mol_graph.SlogP_VSA10
        self_feats[i, 7] = mol_graph.fr_unbrch_alkane
        self_feats[i, 8] = mol_graph.SMR_VSA10
        self_feats[i, 9] = mol_graph.PEOE_VSA12
        # 11
        self_feats[i, 10] = mol_graph.fr_guanido
        self_feats[i, 11] = mol_graph.FpDensityMorgan1
        self_feats[i, 12] = mol_graph.NHOHCount
        self_feats[i, 13] = mol_graph.fr_sulfide
        self_feats[i, 14] = mol_graph.VSA_EState5
        # 16
        self_feats[i, 15] = mol_graph.fr_HOCCN
        self_feats[i, 16] = mol_graph.fr_piperdine
        self_feats[i, 17] = mol_graph.NumSaturatedCarbocycles
        self_feats[i, 18] = mol_graph.fr_amidine
        self_feats[i, 19] = mol_graph.NumHDonors
        # 21
        self_feats[i, 20] = mol_graph.NumAromaticRings
        self_feats[i, 21] = mol_graph.BalabanJ
        self_feats[i, 22] = mol_graph.NumAromaticHeterocycles
        self_feats[i, 23] = mol_graph.MinEStateIndex
        self_feats[i, 24] = mol_graph.fr_Ar_N

        smiles_list.append(mol_graph.smiles)
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    scaffold_ids, le= get_scaffold_id_map(smiles_list)
    
    return (
        batched_graph, 
        torch.tensor(self_feats).to(device), 
        torch.tensor(labels, dtype=torch.float32).to(device),
        torch.tensor(scaffold_ids, dtype=torch.long).to(device),
        smiles_list)

# Vapor Pressure
def descriptor_selection_vp(samples):
    self_feats = np.empty((len(samples), 23), dtype=np.float32)
    smiles_list=[]
    scaffold_ids = np.empty((len(samples),), dtype=np.int64)

    graphs = []
    labels = []
    for i in range(0, len(samples)):
        # ✅ samples[i]가 2개일 수도 있고(테스트), 3개일 수도 있음(훈련)
        if len(samples[i]) == 3:
            mol_graph, label, scaf_id = samples[i]
        else:
            mol_graph, label = samples[i]
            scaf_id = -1  # test용 더미
        # 1
        self_feats[i, 0] = mol_graph.MolMR
        self_feats[i, 1] = mol_graph.TPSA
        self_feats[i, 2] = mol_graph.fr_halogen
        self_feats[i, 3] = mol_graph.SlogP_VSA12
        self_feats[i, 4] = mol_graph.RingCount
        # 6
        self_feats[i, 5] = mol_graph.Kappa1
        self_feats[i, 6] = mol_graph.NumHAcceptors
        self_feats[i, 7] = mol_graph.NumHDonors
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.SMR_VSA5
        # 11
        self_feats[i, 10] = mol_graph.Chi1
        self_feats[i, 11] = mol_graph.Chi3n
        self_feats[i, 12] = mol_graph.BertzCT
        self_feats[i, 13] = mol_graph.VSA_EState8
        self_feats[i, 14] = mol_graph.NumAliphaticCarbocycles
        # 16
        self_feats[i, 15] = mol_graph.HallKierAlpha
        self_feats[i, 16] = mol_graph.VSA_EState6
        self_feats[i, 17] = mol_graph.NumAromaticRings
        self_feats[i, 18] = mol_graph.Chi4n
        self_feats[i, 19] = mol_graph.PEOE_VSA7
        # 21
        self_feats[i, 20] = mol_graph.SlogP_VSA5
        self_feats[i, 21] = mol_graph.VSA_EState7
        self_feats[i, 22] = mol_graph.NOCount

        smiles_list.append(mol_graph.smiles)
        graphs.append(mol_graph)
        labels.append(label)
        scaffold_ids[i] = scaf_id

    batched_graph = dgl.batch(graphs)

    return (
        batched_graph, 
        torch.tensor(self_feats).to(device), 
        torch.tensor(labels, dtype=torch.float32).to(device),
        torch.tensor(scaffold_ids, dtype=torch.long).to(device),
        smiles_list)

# Solubility
def descriptor_selection_solubility(samples):
    self_feats = np.empty((len(samples), 30), dtype=np.float32)
    smiles_list=[]
    scaffold_ids = np.empty((len(samples),), dtype=np.int64)

    graphs = []
    labels = []
    for i in range(0, len(samples)):
        # ✅ samples[i]가 2개일 수도 있고(테스트), 3개일 수도 있음(훈련)
        if len(samples[i]) == 3:
            mol_graph, label, scaf_id = samples[i]
        else:
            mol_graph, label = samples[i]
            scaf_id = -1  # test용 더미
        # 1
        self_feats[i, 0] = mol_graph.Chi1v
        self_feats[i, 1] = mol_graph.Chi1
        self_feats[i, 2] = mol_graph.SlogP_VSA2
        self_feats[i, 3] = mol_graph.HallKierAlpha
        self_feats[i, 4] = mol_graph.PEOE_VSA6
        # 6
        self_feats[i, 5] = mol_graph.fr_benzene
        self_feats[i, 6] = mol_graph.BertzCT
        self_feats[i, 7] = mol_graph.VSA_EState6
        self_feats[i, 8] = mol_graph.SMR_VSA7
        self_feats[i, 9] = mol_graph.Chi3n
        # 11
        self_feats[i, 10] = mol_graph.HeavyAtomMolWt
        self_feats[i, 11] = mol_graph.SMR_VSA10
        self_feats[i, 12] = mol_graph.Kappa1
        self_feats[i, 13] = mol_graph.fr_quatN
        self_feats[i, 14] = mol_graph.PEOE_VSA7
        # 16
        self_feats[i, 15] = mol_graph.NumHDonors
        self_feats[i, 16] = mol_graph.MinEStateIndex
        self_feats[i, 17] = mol_graph.fr_C_O_noCOO
        self_feats[i, 18] = mol_graph.EState_VSA1
        self_feats[i, 19] = mol_graph.MolLogP
        # 21
        self_feats[i, 20] = mol_graph.fr_halogen
        self_feats[i, 21] = mol_graph.SlogP_VSA3
        self_feats[i, 22] = mol_graph.SlogP_VSA5
        self_feats[i, 23] = mol_graph.SlogP_VSA1
        self_feats[i, 24] = mol_graph.SlogP_VSA12
        # 26
        self_feats[i, 25] = mol_graph.VSA_EState10
        self_feats[i, 26] = mol_graph.MinPartialCharge
        self_feats[i, 27] = mol_graph.Kappa2
        self_feats[i, 28] = mol_graph.NHOHCount
        self_feats[i, 29] = mol_graph.SlogP_VSA6

        smiles_list.append(mol_graph.smiles)
        graphs.append(mol_graph)
        labels.append(label)
        scaffold_ids[i] = scaf_id

    batched_graph = dgl.batch(graphs)

    return (
        batched_graph, 
        torch.tensor(self_feats).to(device), 
        torch.tensor(labels, dtype=torch.float32).to(device),
        torch.tensor(scaffold_ids, dtype=torch.long).to(device),
        smiles_list)



# Full Descriptors
def full_descriptors(samples):
    self_feats = np.empty((len(samples), 196), dtype=np.float32)
    smiles_list=[]

    for i in range(0, len(samples)):
        mol_graph = samples[i][0]
        # 1
        self_feats[i, 0] = mol_graph.MaxEStateIndex
        self_feats[i, 1] = mol_graph.MinEStateIndex
        self_feats[i, 2] = mol_graph.MaxAbsEStateIndex
        self_feats[i, 3] = mol_graph.MinAbsEStateIndex
        self_feats[i, 4] = mol_graph.qed
        # 6
        self_feats[i, 5] = mol_graph.MolWt
        self_feats[i, 6] = mol_graph.HeavyAtomMolWt
        self_feats[i, 7] = mol_graph.ExactMolWt
        self_feats[i, 8] = mol_graph.NumValenceElectrons
        self_feats[i, 9] = mol_graph.NumRadicalElectrons

        self_feats[i, 10] = mol_graph.FpDensityMorgan1
        self_feats[i, 11] = mol_graph.FpDensityMorgan2
        self_feats[i, 12] = mol_graph.FpDensityMorgan3
        self_feats[i, 13] = mol_graph.BalabanJ
        self_feats[i, 14] = mol_graph.BertzCT

        self_feats[i, 15] = mol_graph.Chi0
        self_feats[i, 16] = mol_graph.Chi0n
        self_feats[i, 17] = mol_graph.Chi0v
        self_feats[i, 18] = mol_graph.Chi1
        self_feats[i, 19] = mol_graph.Chi1n

        self_feats[i, 20] = mol_graph.Chi1v
        self_feats[i, 21] = mol_graph.Chi2n
        self_feats[i, 22] = mol_graph.Chi2v
        self_feats[i, 23] = mol_graph.Chi3n
        self_feats[i, 24] = mol_graph.Chi3v

        self_feats[i, 25] = mol_graph.Chi4n
        self_feats[i, 26] = mol_graph.Chi4v
        self_feats[i, 27] = mol_graph.HallKierAlpha
        self_feats[i, 28] = mol_graph.Ipc
        self_feats[i, 29] = mol_graph.Kappa1

        self_feats[i, 30] = mol_graph.Kappa2
        self_feats[i, 31] = mol_graph.Kappa3
        self_feats[i, 32] = mol_graph.LabuteASA
        self_feats[i, 33] = mol_graph.PEOE_VSA1
        self_feats[i, 34] = mol_graph.PEOE_VSA10

        self_feats[i, 35] = mol_graph.PEOE_VSA11
        self_feats[i, 36] = mol_graph.PEOE_VSA12
        self_feats[i, 37] = mol_graph.PEOE_VSA13
        self_feats[i, 38] = mol_graph.PEOE_VSA14
        self_feats[i, 39] = mol_graph.PEOE_VSA2

        self_feats[i, 40] = mol_graph.PEOE_VSA3
        self_feats[i, 41] = mol_graph.PEOE_VSA4
        self_feats[i, 42] = mol_graph.PEOE_VSA5
        self_feats[i, 43] = mol_graph.PEOE_VSA6
        self_feats[i, 44] = mol_graph.PEOE_VSA7

        self_feats[i, 45] = mol_graph.PEOE_VSA8
        self_feats[i, 46] = mol_graph.PEOE_VSA9
        self_feats[i, 47] = mol_graph.SMR_VSA1
        self_feats[i, 48] = mol_graph.SMR_VSA10
        self_feats[i, 49] = mol_graph.SMR_VSA2

        self_feats[i, 50] = mol_graph.SMR_VSA3
        self_feats[i, 51] = mol_graph.SMR_VSA4
        self_feats[i, 52] = mol_graph.SMR_VSA5
        self_feats[i, 53] = mol_graph.SMR_VSA6
        self_feats[i, 54] = mol_graph.SMR_VSA7

        self_feats[i, 55] = mol_graph.SMR_VSA8
        self_feats[i, 56] = mol_graph.SMR_VSA9
        self_feats[i, 57] = mol_graph.SlogP_VSA1
        self_feats[i, 58] = mol_graph.SlogP_VSA10
        self_feats[i, 59] = mol_graph.SlogP_VSA11

        self_feats[i, 60] = mol_graph.SlogP_VSA12
        self_feats[i, 61] = mol_graph.SlogP_VSA2
        self_feats[i, 62] = mol_graph.SlogP_VSA3
        self_feats[i, 63] = mol_graph.SlogP_VSA4
        self_feats[i, 64] = mol_graph.SlogP_VSA5

        self_feats[i, 65] = mol_graph.SlogP_VSA6
        self_feats[i, 66] = mol_graph.SlogP_VSA7
        self_feats[i, 67] = mol_graph.SlogP_VSA8
        self_feats[i, 68] = mol_graph.SlogP_VSA9
        self_feats[i, 69] = mol_graph.TPSA

        self_feats[i, 70] = mol_graph.EState_VSA1
        self_feats[i, 71] = mol_graph.EState_VSA10
        self_feats[i, 72] = mol_graph.EState_VSA11
        self_feats[i, 73] = mol_graph.EState_VSA2
        self_feats[i, 74] = mol_graph.EState_VSA3

        self_feats[i, 75] = mol_graph.EState_VSA4
        self_feats[i, 76] = mol_graph.EState_VSA5
        self_feats[i, 77] = mol_graph.EState_VSA6
        self_feats[i, 78] = mol_graph.EState_VSA7
        self_feats[i, 79] = mol_graph.EState_VSA8

        self_feats[i, 80] = mol_graph.EState_VSA9
        self_feats[i, 81] = mol_graph.VSA_EState1
        self_feats[i, 82] = mol_graph.VSA_EState10
        self_feats[i, 83] = mol_graph.VSA_EState2
        self_feats[i, 84] = mol_graph.VSA_EState3

        self_feats[i, 85] = mol_graph.VSA_EState4
        self_feats[i, 86] = mol_graph.VSA_EState5
        self_feats[i, 87] = mol_graph.VSA_EState6
        self_feats[i, 88] = mol_graph.VSA_EState7
        self_feats[i, 89] = mol_graph.VSA_EState8

        self_feats[i, 90] = mol_graph.VSA_EState9
        self_feats[i, 91] = mol_graph.FractionCSP3
        self_feats[i, 92] = mol_graph.HeavyAtomCount
        self_feats[i, 93] = mol_graph.NHOHCount
        self_feats[i, 94] = mol_graph.NOCount

        self_feats[i, 95] = mol_graph.NumAliphaticCarbocycles
        self_feats[i, 96] = mol_graph.NumAliphaticHeterocycles
        self_feats[i, 97] = mol_graph.NumAliphaticRings
        self_feats[i, 98] = mol_graph.NumAromaticCarbocycles
        self_feats[i, 99] = mol_graph.NumAromaticHeterocycles

        self_feats[i, 100] = mol_graph.NumAromaticRings
        self_feats[i, 101] = mol_graph.NumHAcceptors
        self_feats[i, 102] = mol_graph.NumHDonors
        self_feats[i, 103] = mol_graph.NumHeteroatoms
        self_feats[i, 104] = mol_graph.NumRotatableBonds

        self_feats[i, 105] = mol_graph.NumSaturatedCarbocycles
        self_feats[i, 106] = mol_graph.NumSaturatedHeterocycles
        self_feats[i, 107] = mol_graph.NumSaturatedRings
        self_feats[i, 108] = mol_graph.RingCount
        self_feats[i, 109] = mol_graph.MolLogP

        self_feats[i, 110] = mol_graph.MolMR
        self_feats[i, 111] = mol_graph.fr_Al_COO
        self_feats[i, 112] = mol_graph.fr_Al_OH
        self_feats[i, 113] = mol_graph.fr_Al_OH_noTert
        self_feats[i, 114] = mol_graph.fr_ArN

        self_feats[i, 115] = mol_graph.fr_Ar_COO
        self_feats[i, 116] = mol_graph.fr_Ar_N
        self_feats[i, 117] = mol_graph.fr_Ar_NH
        self_feats[i, 118] = mol_graph.fr_Ar_OH
        self_feats[i, 119] = mol_graph.fr_COO

        self_feats[i, 120] = mol_graph.fr_COO2
        self_feats[i, 121] = mol_graph.fr_C_O
        self_feats[i, 122] = mol_graph.fr_C_O_noCOO
        self_feats[i, 123] = mol_graph.fr_C_S
        self_feats[i, 124] = mol_graph.fr_HOCCN

        self_feats[i, 125] = mol_graph.fr_Imine
        self_feats[i, 126] = mol_graph.fr_NH0
        self_feats[i, 127] = mol_graph.fr_NH1
        self_feats[i, 128] = mol_graph.fr_NH2
        self_feats[i, 129] = mol_graph.fr_N_O

        self_feats[i, 130] = mol_graph.fr_Ndealkylation1
        self_feats[i, 131] = mol_graph.fr_Ndealkylation2
        self_feats[i, 132] = mol_graph.fr_Nhpyrrole
        self_feats[i, 133] = mol_graph.fr_SH
        self_feats[i, 134] = mol_graph.fr_aldehyde

        self_feats[i, 135] = mol_graph.fr_alkyl_carbamate
        self_feats[i, 136] = mol_graph.fr_alkyl_halide
        self_feats[i, 137] = mol_graph.fr_allylic_oxid
        self_feats[i, 138] = mol_graph.fr_amide
        self_feats[i, 139] = mol_graph.fr_amidine

        self_feats[i, 140] = mol_graph.fr_aniline
        self_feats[i, 141] = mol_graph.fr_aryl_methyl
        self_feats[i, 142] = mol_graph.fr_azide
        self_feats[i, 143] = mol_graph.fr_azo
        self_feats[i, 144] = mol_graph.fr_barbitur

        self_feats[i, 145] = mol_graph.fr_benzene
        self_feats[i, 146] = mol_graph.fr_benzodiazepine
        self_feats[i, 147] = mol_graph.fr_bicyclic
        self_feats[i, 148] = mol_graph.fr_diazo
        self_feats[i, 149] = mol_graph.fr_dihydropyridine

        self_feats[i, 150] = mol_graph.fr_epoxide
        self_feats[i, 151] = mol_graph.fr_ester
        self_feats[i, 152] = mol_graph.fr_ether
        self_feats[i, 153] = mol_graph.fr_furan
        self_feats[i, 154] = mol_graph.fr_guanido

        self_feats[i, 155] = mol_graph.fr_halogen
        self_feats[i, 156] = mol_graph.fr_hdrzine
        self_feats[i, 157] = mol_graph.fr_hdrzone
        self_feats[i, 158] = mol_graph.fr_imidazole
        self_feats[i, 159] = mol_graph.fr_imide

        self_feats[i, 160] = mol_graph.fr_isocyan
        self_feats[i, 161] = mol_graph.fr_isothiocyan
        self_feats[i, 162] = mol_graph.fr_ketone
        self_feats[i, 163] = mol_graph.fr_ketone_Topliss
        self_feats[i, 164] = mol_graph.fr_lactam

        self_feats[i, 165] = mol_graph.fr_lactone
        self_feats[i, 166] = mol_graph.fr_methoxy
        self_feats[i, 167] = mol_graph.fr_morpholine
        self_feats[i, 168] = mol_graph.fr_nitrile
        self_feats[i, 169] = mol_graph.fr_nitro

        self_feats[i, 170] = mol_graph.fr_nitro_arom
        self_feats[i, 171] = mol_graph.fr_nitro_arom_nonortho
        self_feats[i, 172] = mol_graph.fr_nitroso
        self_feats[i, 173] = mol_graph.fr_oxazole
        self_feats[i, 174] = mol_graph.fr_oxime

        self_feats[i, 175] = mol_graph.fr_para_hydroxylation
        self_feats[i, 176] = mol_graph.fr_phenol
        self_feats[i, 177] = mol_graph.fr_phenol_noOrthoHbond
        self_feats[i, 178] = mol_graph.fr_phos_acid
        self_feats[i, 179] = mol_graph.fr_phos_ester

        self_feats[i, 180] = mol_graph.fr_piperdine
        self_feats[i, 181] = mol_graph.fr_piperzine
        self_feats[i, 182] = mol_graph.fr_priamide
        self_feats[i, 183] = mol_graph.fr_prisulfonamd
        self_feats[i, 184] = mol_graph.fr_pyridine

        self_feats[i, 185] = mol_graph.fr_quatN
        self_feats[i, 186] = mol_graph.fr_sulfide
        self_feats[i, 187] = mol_graph.fr_sulfonamd
        self_feats[i, 188] = mol_graph.fr_sulfone
        self_feats[i, 189] = mol_graph.fr_term_acetylene

        self_feats[i, 190] = mol_graph.fr_tetrazole
        self_feats[i, 191] = mol_graph.fr_thiazole
        self_feats[i, 192] = mol_graph.fr_thiocyan
        self_feats[i, 193] = mol_graph.fr_thiophene
        self_feats[i, 194] = mol_graph.fr_unbrch_alkane

        self_feats[i, 195] = mol_graph.fr_urea

        # 11
        # self_feats[i, 10] = mol_graph.MaxPartialCharge
        # self_feats[i, 11] = mol_graph.MinPartialCharge
        # self_feats[i, 12] = mol_graph.MaxAbsPartialCharge
        # self_feats[i, 13] = mol_graph.MinAbsPartialCharge
        # self_feats[i, 17] = mol_graph.BCUT2D_MWHI
        # self_feats[i, 18] = mol_graph.BCUT2D_MWLOW
        # self_feats[i, 19] = mol_graph.BCUT2D_CHGHI
        # 21
        # self_feats[i, 20] = mol_graph.BCUT2D_CHGLO
        # self_feats[i, 21] = mol_graph.BCUT2D_LOGPHI
        # self_feats[i, 22] = mol_graph.BCUT2D_LOGPLOW
        # self_feats[i, 23] = mol_graph.BCUT2D_MRHI
        # self_feats[i, 24] = mol_graph.BCUT2D_MRLOW

        smiles_list.append(mol_graph.smiles)

    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    return (
        batched_graph, 
        torch.tensor(self_feats).to(device), 
        torch.tensor(labels, dtype=torch.float32).to(device),
        smiles_list)


# # Debug
# def descriptor_selection_debug(samples):

#     descriptor_names = [
#             "MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex", "qed",
#             "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumRadicalElectrons",
#             "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge", "FpDensityMorgan1",
#             "FpDensityMorgan2", "FpDensityMorgan3", "BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI",
#             "BCUT2D_CHGLO", "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW",
#             "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v",
#             "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
#             "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha",
#             "Ipc", "Kappa1", "Kappa2", "Kappa3", "LabuteASA",
#             "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13",
#             "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
#             "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1",
#             "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5",
#             "SMR_VSA6", "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1",
#             "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3",
#             "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
#             "SlogP_VSA9", "TPSA", "EState_VSA1", "EState_VSA10", "EState_VSA11",
#             "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6",
#             "EState_VSA7", "EState_VSA8", "EState_VSA9", "VSA_EState1", "VSA_EState10",
#             "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6",
#             "VSA_EState7", "VSA_EState8", "VSA_EState9", "FractionCSP3", "HeavyAtomCount",
#             "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
#             "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors",
#             "NumHeteroatoms", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings",
#             "RingCount", "MolLogP", "MolMR", "fr_Al_COO", "fr_Al_OH",
#             "fr_Al_OH_noTert", "fr_ArN", "fr_Ar_COO", "fr_Ar_N", "fr_Ar_NH",
#             "fr_Ar_OH", "fr_COO", "fr_COO2", "fr_C_O", "fr_C_O_noCOO",
#             "fr_C_S", "fr_HOCCN", "fr_Imine", "fr_NH0", "fr_NH1",
#             "fr_NH2", "fr_N_O", "fr_Ndealkylation1", "fr_Ndealkylation2", "fr_Nhpyrrole",
#             "fr_SH", "fr_aldehyde", "fr_alkyl_carbamate", "fr_alkyl_halide", "fr_allylic_oxid",
#             "fr_amide", "fr_amidine", "fr_aniline", "fr_aryl_methyl", "fr_azide",
#             "fr_azo", "fr_barbitur", "fr_benzene", "fr_benzodiazepine", "fr_bicyclic",
#             "fr_diazo", "fr_dihydropyridine", "fr_epoxide", "fr_ester", "fr_ether",
#             "fr_furan", "fr_guanido", "fr_halogen", "fr_hdrzine", "fr_hdrzone",
#             "fr_imidazole", "fr_imide", "fr_isocyan", "fr_isothiocyan", "fr_ketone",
#             "fr_ketone_Topliss", "fr_lactam", "fr_lactone", "fr_methoxy", "fr_morpholine",
#             "fr_nitrile", "fr_nitro", "fr_nitro_arom", "fr_nitro_arom_nonortho", "fr_nitroso",
#             "fr_oxazole", "fr_oxime", "fr_para_hydroxylation", "fr_phenol", "fr_phenol_noOrthoHbond",
#             "fr_phos_acid", "fr_phos_ester", "fr_piperdine", "fr_piperzine", "fr_priamide",
#             "fr_prisulfonamd", "fr_pyridine", "fr_quatN", "fr_sulfide", "fr_sulfonamd",
#             "fr_sulfone", "fr_term_acetylene", "fr_tetrazole", "fr_thiazole", "fr_thiocyan",
#             "fr_thiophene", "fr_unbrch_alkane", "fr_urea"
#         ]

#     self_feats = np.empty((len(samples), len(descriptor_names)), dtype=np.float32)
#     smiles_list = []

#     for i in range(len(samples)):

#         mol_graph = samples[i][0]
#         smiles = mol_graph.smiles
#         smiles_list.append(smiles)

#         # descriptor values 한번에 리스트화
#         descriptor_values =[
#     mol_graph.MaxEStateIndex, mol_graph.MinEStateIndex, mol_graph.MaxAbsEStateIndex, mol_graph.MinAbsEStateIndex, mol_graph.qed,
#     mol_graph.MolWt, mol_graph.HeavyAtomMolWt, mol_graph.ExactMolWt, mol_graph.NumValenceElectrons, mol_graph.NumRadicalElectrons,
#     mol_graph.MaxPartialCharge, mol_graph.MinPartialCharge, mol_graph.MaxAbsPartialCharge, mol_graph.MinAbsPartialCharge, mol_graph.FpDensityMorgan1,
#     mol_graph.FpDensityMorgan2, mol_graph.FpDensityMorgan3, mol_graph.BCUT2D_MWHI, mol_graph.BCUT2D_MWLOW, mol_graph.BCUT2D_CHGHI,
#     mol_graph.BCUT2D_CHGLO, mol_graph.BCUT2D_LOGPHI, mol_graph.BCUT2D_LOGPLOW, mol_graph.BCUT2D_MRHI, mol_graph.BCUT2D_MRLOW,
#     mol_graph.BalabanJ, mol_graph.BertzCT, mol_graph.Chi0, mol_graph.Chi0n, mol_graph.Chi0v,
#     mol_graph.Chi1, mol_graph.Chi1n, mol_graph.Chi1v, mol_graph.Chi2n, mol_graph.Chi2v,
#     mol_graph.Chi3n, mol_graph.Chi3v, mol_graph.Chi4n, mol_graph.Chi4v, mol_graph.HallKierAlpha,
#     mol_graph.Ipc, mol_graph.Kappa1, mol_graph.Kappa2, mol_graph.Kappa3, mol_graph.LabuteASA,
#     mol_graph.PEOE_VSA1, mol_graph.PEOE_VSA10, mol_graph.PEOE_VSA11, mol_graph.PEOE_VSA12, mol_graph.PEOE_VSA13,
#     mol_graph.PEOE_VSA14, mol_graph.PEOE_VSA2, mol_graph.PEOE_VSA3, mol_graph.PEOE_VSA4, mol_graph.PEOE_VSA5,
#     mol_graph.PEOE_VSA6, mol_graph.PEOE_VSA7, mol_graph.PEOE_VSA8, mol_graph.PEOE_VSA9, mol_graph.SMR_VSA1,
#     mol_graph.SMR_VSA10, mol_graph.SMR_VSA2, mol_graph.SMR_VSA3, mol_graph.SMR_VSA4, mol_graph.SMR_VSA5,
#     mol_graph.SMR_VSA6, mol_graph.SMR_VSA7, mol_graph.SMR_VSA8, mol_graph.SMR_VSA9, mol_graph.SlogP_VSA1,
#     mol_graph.SlogP_VSA10, mol_graph.SlogP_VSA11, mol_graph.SlogP_VSA12, mol_graph.SlogP_VSA2, mol_graph.SlogP_VSA3,
#     mol_graph.SlogP_VSA4, mol_graph.SlogP_VSA5, mol_graph.SlogP_VSA6, mol_graph.SlogP_VSA7, mol_graph.SlogP_VSA8,
#     mol_graph.SlogP_VSA9, mol_graph.TPSA, mol_graph.EState_VSA1, mol_graph.EState_VSA10, mol_graph.EState_VSA11,
#     mol_graph.EState_VSA2, mol_graph.EState_VSA3, mol_graph.EState_VSA4, mol_graph.EState_VSA5, mol_graph.EState_VSA6,
#     mol_graph.EState_VSA7, mol_graph.EState_VSA8, mol_graph.EState_VSA9, mol_graph.VSA_EState1, mol_graph.VSA_EState10,
#     mol_graph.VSA_EState2, mol_graph.VSA_EState3, mol_graph.VSA_EState4, mol_graph.VSA_EState5, mol_graph.VSA_EState6,
#     mol_graph.VSA_EState7, mol_graph.VSA_EState8, mol_graph.VSA_EState9, mol_graph.FractionCSP3, mol_graph.HeavyAtomCount,
#     mol_graph.NHOHCount, mol_graph.NOCount, mol_graph.NumAliphaticCarbocycles, mol_graph.NumAliphaticHeterocycles, mol_graph.NumAliphaticRings,
#     mol_graph.NumAromaticCarbocycles, mol_graph.NumAromaticHeterocycles, mol_graph.NumAromaticRings, mol_graph.NumHAcceptors, mol_graph.NumHDonors,
#     mol_graph.NumHeteroatoms, mol_graph.NumRotatableBonds, mol_graph.NumSaturatedCarbocycles, mol_graph.NumSaturatedHeterocycles, mol_graph.NumSaturatedRings,
#     mol_graph.RingCount, mol_graph.MolLogP, mol_graph.MolMR, mol_graph.fr_Al_COO, mol_graph.fr_Al_OH,
#     mol_graph.fr_Al_OH_noTert, mol_graph.fr_ArN, mol_graph.fr_Ar_COO, mol_graph.fr_Ar_N, mol_graph.fr_Ar_NH,
#     mol_graph.fr_Ar_OH, mol_graph.fr_COO, mol_graph.fr_COO2, mol_graph.fr_C_O, mol_graph.fr_C_O_noCOO,
#     mol_graph.fr_C_S, mol_graph.fr_HOCCN, mol_graph.fr_Imine, mol_graph.fr_NH0, mol_graph.fr_NH1,
#     mol_graph.fr_NH2, mol_graph.fr_N_O, mol_graph.fr_Ndealkylation1, mol_graph.fr_Ndealkylation2, mol_graph.fr_Nhpyrrole,
#     mol_graph.fr_SH, mol_graph.fr_aldehyde, mol_graph.fr_alkyl_carbamate, mol_graph.fr_alkyl_halide, mol_graph.fr_allylic_oxid,
#     mol_graph.fr_amide, mol_graph.fr_amidine, mol_graph.fr_aniline, mol_graph.fr_aryl_methyl, mol_graph.fr_azide,
#     mol_graph.fr_azo, mol_graph.fr_barbitur, mol_graph.fr_benzene, mol_graph.fr_benzodiazepine, mol_graph.fr_bicyclic,
#     mol_graph.fr_diazo, mol_graph.fr_dihydropyridine, mol_graph.fr_epoxide, mol_graph.fr_ester, mol_graph.fr_ether,
#     mol_graph.fr_furan, mol_graph.fr_guanido, mol_graph.fr_halogen, mol_graph.fr_hdrzine, mol_graph.fr_hdrzone,
#     mol_graph.fr_imidazole, mol_graph.fr_imide, mol_graph.fr_isocyan, mol_graph.fr_isothiocyan, mol_graph.fr_ketone,
#     mol_graph.fr_ketone_Topliss, mol_graph.fr_lactam, mol_graph.fr_lactone, mol_graph.fr_methoxy, mol_graph.fr_morpholine,
#     mol_graph.fr_nitrile, mol_graph.fr_nitro, mol_graph.fr_nitro_arom, mol_graph.fr_nitro_arom_nonortho, mol_graph.fr_nitroso,
#     mol_graph.fr_oxazole, mol_graph.fr_oxime, mol_graph.fr_para_hydroxylation, mol_graph.fr_phenol, mol_graph.fr_phenol_noOrthoHbond,
#     mol_graph.fr_phos_acid, mol_graph.fr_phos_ester, mol_graph.fr_piperdine, mol_graph.fr_piperzine, mol_graph.fr_priamide,
#     mol_graph.fr_prisulfonamd, mol_graph.fr_pyridine, mol_graph.fr_quatN, mol_graph.fr_sulfide, mol_graph.fr_sulfonamd,
#     mol_graph.fr_sulfone, mol_graph.fr_term_acetylene, mol_graph.fr_tetrazole, mol_graph.fr_thiazole, mol_graph.fr_thiocyan,
#     mol_graph.fr_thiophene, mol_graph.fr_unbrch_alkane, mol_graph.fr_urea
# ]

#         # ===== NaN 체크 =====
#         for j, val in enumerate(descriptor_values):
#             if np.isnan(val):
#                 print(f"[NaN DETECTED] sample_idx={i}")
#                 print(f"SMILES: {smiles}")
#                 print(f"Descriptor: {descriptor_names[j]}")
#                 print("-" * 50)

#         self_feats[i] = descriptor_values

#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)

#     return (
#         batched_graph,
#         torch.tensor(self_feats).to(device),
#         torch.tensor(labels, dtype=torch.float32).to(device),
#         smiles_list
#     )