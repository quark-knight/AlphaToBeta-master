
# imports
'''
this file is meant to store all the functions meant for mutating the giving the protein 
'''
from biopandas.pdb import PandasPdb
import numpy as np 

# Amino acid mappings to used later if needed
# Fixed 20-aa order
AA_ORDER = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'P']

# AA_ORDER = ['A','R','N','D','C','Q','E','G','H','I',
#             'L','K','M','F','P','S','T','W','Y','V']

AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ORDER)}

# 3-letter -> 1-letter map (PDB residue names)
THREE_TO_ONE = {
    # 'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    # 'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    # 'THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I',
    'LYS':'K','LEU':'L','MET':'M','ASN':'N','GLN':'Q','ARG':'R','SER':'S','THR':'T',
    'VAL':'V','TRP':'W','TYR':'Y','PRO':'P'
}

def pdb2df(file_path):
    '''
    this function takes in a pdb file's file path and returns a dataframe for the rows corresponding to "ATOM"

    Parameters
    ----------
    file_path : str
        file path of the input file in PDB format

    Returns
    -------
    A Pandas DataFrame

    '''
    # defining the amino acids
    # In the list below there are only 20 natural amino acids - selenocysteine and pyrroleucine are excluded.

    three_to_one = {
        'ALA': 'A',
        'ARG': 'R',
        'ASN': 'N',
        'ASP': 'D',
        'CYS': 'C',
        'GLN': 'Q',
        'GLU': 'E',
        'GLY': 'G',
        'HIS': 'H',
        'ILE': 'I',
        'LEU': 'L',
        'LYS': 'K',
        'MET': 'M',
        'PHE': 'F',
        'PRO': 'P',
        'SER': 'S',
        'THR': 'T',
        'TRP': 'W',
        'TYR': 'Y',
        'VAL': 'V',
        'CEN': 'Z',
        'NEN': 'X'
    }
    protein = PandasPdb()
    protein.read_pdb(file_path)
    protein_df = protein.df['ATOM']
    protein_df['residue_id'] = protein_df['residue_name'].map(three_to_one)
    return protein_df




def sequence_df(protein_df):
    '''
    takes in the protein DataFrame and outputs a DataFrame containing only residue_number and residue_id

    Parameters
    ----------
    protein_df : Pandas DataFrame Object
    '''
    protein_sequence_df = protein_df[['residue_number','residue_id']].drop_duplicates()
    protein_sequence_df = protein_sequence_df.reset_index()
    protein_sequence_df = protein_sequence_df.drop(columns = 'index')
    return protein_sequence_df




def df2sequence(pdb_df):
    '''
    takes in a sequence DataFrame and outputs a string containing the sequence in string format

    Parameters
    ----------
    pdb_df : Pandas DataFrame Object
    '''
    sequence = ''.join(pdb_df['residue_id'])
    return sequence

def read_pdb_file(file_path):
    '''
    takes in a pdb file and gives back the sequence in string format 
    '''
    chars_to_remove = "ZX"
    input_string = df2sequence(sequence_df(pdb2df(file_path)))
    result_string = ''.join(char for char in input_string if char not in chars_to_remove)
    return result_string


def get_neighbour_table(pdb_path, chain_id='A'):
    '''
    Extract Cα atom coordinates and residue information (1-letter codes) for each residue in a specific chainfrom a PDB file.
    Args:
        pdb_path (str): Path to the PDB file.
        chain_id (str): Chain identifier in the PDB file.
    Returns:
            sub (pd.dataframe): A pandas DataFrame with one row per residue in the chosen chain with columns:
                res_num((int, 1-based)) -- residue number (as in the PDB, not renumbered)
                one (str)              -- 1-letter amino-acid code
                x, y, z (float)        -- Cartesian coordinates of the Cα atom
    If multiple CA atoms for the same residue number exist, only the first is kept. Sorted by residue number.
    '''
    
    ## NOTE: A pdb file contains lines like this ,
        # ATOM      357  CA ALA   A     1  45    11.104  13.207   9.657  1.00 20.00           N
        # where the columns are:
        # Record name   "ATOM  ", atom_number, atom_name, residue_name, chain_id, residue_number, x, y, z coordinates, occupancy, b_factor, element

    ppdb = PandasPdb().read_pdb(pdb_path)     # Read PDB using biopandas
    atom = ppdb.df['ATOM']                    # Getting ATOM dataframe

    # Filter chain & Cα, keeps only alpha-carbon atoms from the requested chain
    sub = atom[(atom['chain_id'] == chain_id) & (atom['atom_name'] == 'CA')].copy()

    # Map residue names to 1-letter, drop non-std residues
    sub['one'] = sub['residue_name'].map(THREE_TO_ONE)
    # '~' negates the selection of isna() function, so it keeps only the rows where 'one' is NOT NaN
    sub = sub[~sub['one'].isna()].copy()

    # Keep the first CA if duplicate entries exist, select relevant coloms and rename columns (we can do these operations in seperate steps too)
    sub = (sub.drop_duplicates(subset=['residue_number'])
           .loc[:, ['residue_number', 'one', 'x_coord', 'y_coord', 'z_coord']]
           .rename(columns={'residue_number':'res_num', 'x_coord':'x','y_coord':'y','z_coord':'z'}))

    # Sort by residue number
    sub = sub.sort_values('res_num').reset_index(drop=True)
    return sub

def get_neibouring_residues(pdb_path,
                           start_idx_0based,
                           end_idx_0based,
                           cutoff_angstrom=6.0,
                           chain_id='A',
                           exclude_segment=True):
    '''
    Count amino-acid frequencies within cutoff of any residue in the mutated segment.
    Args:
        pdb_path            (str):  Path to PDB file.
        start_idx_0based    (int): segment starting index in 0-based scheme (function's convention).
        end_idx_0based      (int): Segment ending index in 0-based scheme (function's convention).
        cutoff_angstrom     (float): Distance threshold (Cα–Cα Euclidean distance).
        chain_id            (str): Chain identifier in the PDB file.
        exclude_segment     (bool: If True, do NOT count residues that are inside the mutated segment itself.
    Returns:
        aa_counts (np.ndarray shape (20,)): Array of amino-acid counts within cutoff, in fixed 20-aa order.
    0 if no residues found or segment invalid.
    '''

    # Get Cα table, if the table is empty, return zeros
    table = get_neighbour_table(pdb_path, chain_id=chain_id)
    if table.empty:
        return np.zeros(20, dtype=int)

    # Segment mapped to PDB 1-based numbering
    seg_min = start_idx_0based + 1
    seg_max = end_idx_0based + 1

    # Split segment CA and non-segment CA
    is_seg = (table['res_num'] >= seg_min) & (table['res_num'] <= seg_max)
    seg = table[is_seg]
    env = table[~is_seg] if exclude_segment else table

    if seg.empty:
        # if segment has no residues (invalid segment), return zeros
        return np.zeros(20, dtype=int)

    # Get coordinates of CA as numpy arrays
    seg_xyz = seg[['x','y','z']].to_numpy(dtype=float)   # (Ns, 3)
    env_xyz = env[['x','y','z']].to_numpy(dtype=float)   # (Ne, 3)

    # Brute-force distances: (Ne, Ns)
    # For typical protein sizes this is fine; if we need speedup, we have to add a KD-tree.
    # This is using numpy broadcasting to compute pairwise distances, broadcasting rules:
    # env_xyz[:, None, :] has shape (Ne, 1, 3)
    # seg_xyz[None, :, :] has shape (1, Ns, 3)
    # The subtraction results diff has shape (Ne, Ns, 3)
    diff = env_xyz[:, None, :] - seg_xyz[None, :, :]
    d2 = np.einsum('nij,nij->ni', diff, diff)   # squared distances per env residue to every seg residue
    within_cutoff_env_residues = (d2 <= cutoff_angstrom**2).any(axis=1)  # .any(any(axis=1) implies for each environment residue, is it close to any segment residue? 

    # Count amino acids among env residues that are within cutoff
    aa_counts = np.zeros(20, dtype=int)
    # Get 1-letter codes of close residues
    within_cutoff_env_residues = env.loc[within_cutoff_env_residues, 'one'].to_numpy()

    # Faster way using np.bincount with a lookup table:
    # Build a byte → index table (lookup), default -1 for unknown AAs
    lookup = np.full(256, -1, dtype=int)
    for aa, idx in AA_TO_IDX.items():
        lookup[ord(aa)] = idx

    # Convert characters in close_env into their byte codes (0–255), then use lookup to get indices (0–19 or -1)
    idxs = lookup[np.frombuffer(within_cutoff_env_residues.astype("S1"), dtype=np.uint8)]
    idxs = idxs[idxs >= 0] # keep only valid indices
    aa_counts = np.bincount(idxs, minlength=20) # np.bincount to count how often each amino acid appears


    # This loop fills the aa_counts array [Old code]
    # for one in within_cutoff_env_residues:
    #     # this looks up the AA_TO_IDX ordered dictionary to get the index of the Amino acid whose single letter code is 'one'
    #     idx = AA_TO_IDX.get(one, None) 
    #     if idx is not None:
    #         aa_counts[idx] += 1
    # Alt way using np.bincount[but slower than current method]:
    # idxs = np.array([AA_TO_IDX.get(aa, -1) for aa in close_env], dtype=int)
    # idxs = idxs[idxs >= 0]
    # aa_counts = np.bincount(idxs, minlength=20)

    return aa_counts
