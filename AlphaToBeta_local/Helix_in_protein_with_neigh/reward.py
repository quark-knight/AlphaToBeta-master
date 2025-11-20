
##################################################################################################
# This is modified from the original reward.py file in that fills an array with neighbouring
# amino acids if the mutation results into a beta sheet 
# The reward function is also modified to give reward based on the percentage of helix vs sheet
#################################################################################################

import torch
import os
import numpy as np 
import biotite.structure as struc # biotite is used for secondary structure annotation
import biotite.structure.io as strucio # for reading the PDB files
from transformers import AutoTokenizer, EsmForProteinFolding  # ESM model from huggingface
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein # OpenFold utils
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37 # OpenFold utils
from datetime import datetime # to give unique names to files
from biopandas.pdb import PandasPdb # to read the PDB files and get the B-factors
from Helix_in_protein.sequence import * 

# Amino acid mappings to used later if needed
# Fixed 20-aa order
AA_ORDER = ['A','R','N','D','C','Q','E','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V']
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# 3-letter -> 1-letter map (PDB residue names)
THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

# DEFINING THE MODEL FOR PROTEIN MODELLING
torch.backends.cuda.matmul.allow_tf32 = True # Allow TF32 on matmul 
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1") # ESM tokenizer 
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1") # ESM model for protein folding

model = model.cuda()
def convert_outputs_to_pdb(outputs):
    """
    Convert model outputs to PDB format strings

    Args:
        outputs (dict): Folding Model outputs containing atom positions, dict[keys: "positions", "aatype", "atom37_atom_exists", "residue_index", "plddt", "chain_index"].
    """
    # Convert 14-atom representation to 37-atom representation, ["positions"][-1] to get the final, most refined structure
    # outputs["positions"][-1] gives a dict with shape (batch size, Number of residues, atom37 representation, 3)
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs) 

    # Move tensors to CPU and convert to numpy arrays
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []

    # Iterate over each protein in the batch
    for i in range(outputs["aatype"].shape[0]):     # batch size
        aa = outputs["aatype"][i]                   # (N_res,)
        pred_pos = final_atom_positions[i]          # (N_res, 37, 3)
        mask = final_atom_mask[i]                   # (N_res, 37)
        resid = outputs["residue_index"][i] + 1     # (N_res,) +1 to convert to 1-indexed
        b_fact = outputs["plddt"][i]                # (N_res,)
        chain_id = outputs["chain_index"][i] if "chain_index" in outputs else None # (N_res,) or None
        # Create OpenFold Protein object
        pred = OFProtein( 
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors= b_fact,
            chain_index=chain_id,
        )
        pdbs.append(to_pdb(pred)) # Convert to PDB format string (contains the HEADER, MODEL, TER, END too along with atoms
    return pdbs

# In[4]:


def generate_structure_from_sequence(sequence,name=None):

    '''
    This function takes in the sequence of a protein and gives back the structure - this is the function where ESM Model is being used
    Args:
        sequence (str): Amino acid sequence of the protein.
        name (str): Name to save the PDB file as. If None, the PDB will not be saved to a file.
    Returns:
        None: The function saves the PDB file to the specified name.
    '''
    # tokenizing the input sequence (one sequence at a time), outputs a dictionary with input_ids and attention_mask (required if padding is true) as keys
    tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'] # (1, sequence length)
    # moving the input to GPU
    tokenized_input = tokenized_input.cuda()

    # no need to compute gradients, we are just doing inference
    with torch.no_grad():               
        output = model(tokenized_input) 
        # getting the model output, which is a dictionary with keys:  "positions", "aatype", "atom37_atom_exists", "residue_index", "plddt", "chain_index"


    pdb = convert_outputs_to_pdb(output)
    with open(f"{name}.pdb", "w") as f:
        f.write("".join(pdb))
    

def get_structural_annotations(file_path)-> np.ndarray:
    """
    This function takes in a file path to a PDB file and gives back the secondary structure annotations using biotite package
    Args:
        file_path (str): Path to the PDB file generated by ESM model.
    Returns:
        np.ndarray: Array of secondary structure annotations from biotite package.
    """
    array = strucio.load_structure(file_path)
    sse = struc.annotate_sse(array, chain_id="A")
    return sse

def percentage_of_secondary_structure(arr,secondary_structure_type,starting_residue,ending_residue):
        
    '''
    This function is used to find the percentage secondary stuctural type in a peptide within a protein
    Args:
        arr (np.ndarray): Array of secondary structure annotations from biotite package.
        secondary_structure_type (str): Type of secondary structure to calculate percentage for. Options are 'helix', 'sheet', or 'both'.
        starting_residue (int): Starting residue index (0-based) for the segment of interest.
        ending_residue (int): Ending residue index (0-based) for the segment of interest.
    Returns:
        (float or tuple): Percentage of the specified secondary structure type(s) in the segment.
    '''
    
    if len(arr) == 0:
        raise ValueError("Input array must not be empty")
    # this will count the number of amino acids that are part of a helix.
    # print(starting_residue,ending_residue)
    # print(len(arr))
    arr = arr[starting_residue:ending_residue+1] # +1 because ending residue is inclusive   
    total_elements = len(arr)
    count_a = np.count_nonzero(arr == 'a')
    percentage_a = (count_a / total_elements) * 100

    # this will compute the number of amino acids that are part of a sheet. 
    count_b = np.count_nonzero(arr == 'b')
    percentage_b = (count_b / total_elements) * 100

    # if secondary_structure_type == 'helix':
    #     return percentage_a

    # if secondary_structure_type == 'sheet':
    #     return percentage_b

    if secondary_structure_type == 'both':
        return percentage_a, percentage_b



def give_time_as_string():
    '''
    This function gives the current time as a string without spaces, useful for giving unique names to files
    Returns:
        str: Current time formatted as a string without spaces.
    '''
    current_time = datetime.now()
    # Format the current time as a string without spaces
    time_str = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    return time_str

def plddt_value_of_helical_residues(structure_path, starting_residue, ending_residue):
    '''
    This function takes in the path to a PDB file and gives back the fraction of residues in the specified range that have a pLDDT value >= 0.7
    Args:
        structure_path (str): Path to the PDB file.
        starting_residue (int): Starting residue index (0-based) for the segment of interest.
        ending_residue (int): Ending residue index (0-based) for the segment of interest.
    Returns:
        float: Fraction of residues in the specified range with pLDDT >= 0.7
    '''
    # reading the PDB file using biopandas
    ppdb = PandasPdb().read_pdb(structure_path)
    df = ppdb.df['ATOM']
    condition = (df['residue_number']>=starting_residue+1) & (df['residue_number']<=ending_residue+1)
    df = df[condition] # getting the rows corresponding to the interesting helix
    average_b_factors = df.groupby('residue_number')['b_factor'].mean().reset_index()
    threshold = 0.7 #?????????/ this cutoff is chosen based on alphafold website (but we are using from ESMFold which is a different model, so this might need to be changed)
    fraction_with_acceptable_plddt = len(average_b_factors[average_b_factors['b_factor']>= threshold])/len(average_b_factors)
    return fraction_with_acceptable_plddt


def get_ca_table(pdb_path, chain_id='A'):
    """
    Return per-residue table with CA coords and 1-letter codes.
    Columns: resseq (int, 1-based), one (str), x,y,z (float).
    Only ATOM, only given chain_id, only CA records.
    """
    ppdb = PandasPdb().read_pdb(pdb_path)
    atom = ppdb.df['ATOM']

    # Filter chain & CA
    sub = atom[(atom['chain_id'] == chain_id) & (atom['atom_name'] == 'CA')].copy()

    # Map residue names to 1-letter, drop non-std residues
    sub['one'] = sub['residue_name'].map(THREE_TO_ONE)
    sub = sub[~sub['one'].isna()].copy()

    # Keep the first CA if duplicate entries exist
    sub = (sub
           .drop_duplicates(subset=['residue_number'])
           .loc[:, ['residue_number', 'one', 'x_coord', 'y_coord', 'z_coord']]
           .rename(columns={'residue_number':'resseq',
                            'x_coord':'x','y_coord':'y','z_coord':'z'}))

    # Sort by residue number
    sub = sub.sort_values('resseq').reset_index(drop=True)
    return sub

def count_aa_within_cutoff(pdb_path,
                           start_idx_0based,
                           end_idx_0based,
                           cutoff_angstrom=6.0,
                           chain_id='A',
                           exclude_segment=True):
    """
    Count amino-acid frequencies within cutoff of any residue in the mutated segment.

    Parameters
    ----------
    start_idx_0based, end_idx_0based : int
        Segment indices in 0-based scheme (your function's convention).
    cutoff_angstrom : float
        Distance threshold (Cα–Cα Euclidean distance).
    exclude_segment : bool
        If True, do NOT count residues that are inside the mutated segment itself.

    Returns
    -------
    np.ndarray shape (20,)
        Counts in AA_ORDER order.
    """
    table = get_ca_table(pdb_path, chain_id=chain_id)
    if table.empty:
        return np.zeros(20, dtype=int)

    # Segment mapped to PDB 1-based numbering
    seg_min = start_idx_0based + 1
    seg_max = end_idx_0based + 1

    # Split segment CA and non-segment CA
    is_seg = (table['resseq'] >= seg_min) & (table['resseq'] <= seg_max)
    seg = table[is_seg]
    env = table[~is_seg] if exclude_segment else table

    if seg.empty:
        # If window invalid, just return zeros
        return np.zeros(20, dtype=int)

    seg_xyz = seg[['x','y','z']].to_numpy(dtype=float)   # (Ns, 3)
    env_xyz = env[['x','y','z']].to_numpy(dtype=float)   # (Ne, 3)

    # Brute-force distances: (Ne, Ns)
    # For typical protein sizes this is fine; if we need speedup, we have to add a KD-tree.
    diff = env_xyz[:, None, :] - seg_xyz[None, :, :]
    d2 = np.einsum('nij,nij->ni', diff, diff)   # squared distances per env residue to every seg residue
    within = (d2 <= cutoff_angstrom**2).any(axis=1)  # any seg residue close

    # Count amino acids among env residues that are within cutoff
    aa_counts = np.zeros(20, dtype=int)
    close_env = env.loc[within, 'one'].to_numpy()

    for one in close_env:
        idx = AA_TO_IDX.get(one, None)
        if idx is not None:
            aa_counts[idx] += 1

    return aa_counts



def get_reward_from_result(result_pct_got, result_plddt,cutoff=70,usage_of_plddt=False):
    '''
    Decide reward based on helix vs sheet content. It takes in a reward cutoff and gives back the reward, 
    after looking at the result, which is the percentage content of the secondary structure. 

    Parameters
    ----------
    result_pct_got : float or tuple
        Percentage of the specified secondary structure type in the segment.
    result_pct_got[0] : float
        Percentage of helix residues in the segment (0-100).
    result_pct_got[0] : float
        Percentage of sheet residues in the segment (0-100).
    cutoff : float, default=30
        Threshold for "too much sheet" / Reward cutoff percentage
    usage_of_plddt : bool
        Whether to consider pLDDT in the reward calculation.
    result_plddt : float
    Fraction of residues in the specified range with pLDDT >= 0.7
        NOTE: if helix_pct/sheet_pct are in 0-100 scale, set cutoff=70.0  
    Returns
    -------
        float: Reward value based on the criteria.
    '''
    if isinstance(result_pct_got, tuple):
        # result has multiple values (helix %, sheet %)
        helix_pct = result_pct_got[0]
        sheet_pct = result_pct_got[1]
        if usage_of_plddt == False:
            if helix_pct > sheet_pct:
            # Case 1: helix dominates
                return -0.01
            elif sheet_pct > helix_pct and sheet_pct < cutoff:
                # Case 2: sheet dominates but below cutoff
                return 0.01
            elif sheet_pct >= cutoff and sheet_pct > helix_pct:
                # Case 3: sheet dominates and exceeds cutoff
                return 10.0
            else:
                # Edge case: tie or undefined
                return 0.0
            
            # if helix_pct >=cutoff[0] and result_pct_got[1] >= cutoff[1] and result_plddt >= cutoff[2]: # 
            #     return -0.01
            # elif result_pct_got[0] >=cutoff[0] and result_pct_got[1] >= cutoff[1] and result_plddt >= cutoff[2]:
            #     return -0.005
            # else:
            #     return 10
    #     if usage_of_plddt == True:
    #         pass
    # else:
    #     # result is just a single float
    #     if usage_of_plddt == True:
    #         if result_pct_got >=cutoff and result_plddt >= cutoff: # only if both plddt and helical content is greater than threshold. 
    #             return -0.01
    #         else:
    #             return 10
        
    #     if usage_of_plddt == False:
    #         if result_pct_got < cutoff:
    #             return 10
    #         else:
    #             return -0.01
def reward_function_with_env_counts(template_protein_structure_path,
                    protein_sequence,
                    reward_cutoff_sheet,
                    unique_name_to_give,
                    starting_residue_id,
                    ending_residue_id,
                    secondary_structure_type_from_env ='both',
                    validation=False,
                    folder_to_save_validation_files=None,
                    use_plddt = False,
                    distance_cutoff=6.0,
                    chain_id='A',
):
    '''
    This function is used to calculate the reward based on the percentage of a specified secondary structure type in a segment of a protein.
    It generates the structure from the sequence using ESM model, annotates the secondary structure using biotite, and calculates the reward based on the criteria.
    This is called by the (gym) environment class to evaluate the rewards and is the most computationally expensive part
    Args:
        template_protein_structure_path (str): Path to the template protein structure file (used only in validation mode).
        protein_sequence (str): Amino acid sequence of the protein.
        reward_cutoff (float or tuple): Reward cutoff percentage.
        unique_name_to_give (str): Unique name to save the generated PDB file.
        starting_residue_id (int): Starting residue index (0-based) for the segment of interest.
        ending_residue_id (int): Ending residue index (0-based) for the segment of interest.
        secondary_structure_type_from_env (str): Type of secondary structure to calculate percentage for. Options are 'helix', 'sheet', or 'both'. Default is 'helix'.
        validation (bool): Whether to run in validation mode (saves files in a specified folder). Default is False.
        folder_to_save_validation_files (str): Folder path to save validation files (used only in validation mode).
        use_plddt (bool): Whether to consider pLDDT in the reward calculation. Default is False.
        distance_cutoff (float): Distance cutoff in Angstroms to consider neighboring residues. Default is 6.0.
        chain_id (str): Chain identifier in the PDB file. Default is 'A'.

    Returns: (reward, aa_counts_or_none)
        float: Reward value based on the criteria.
        aa_counts_or_none is a length-20 array if sheet dominates, else None.
    '''
    # 1) Predict structure -> write PDB
    if not validation:
        name = f'NEW_{unique_name_to_give}'
        generate_structure_from_sequence(protein_sequence, name=name)
        structure_path = f"{name}.pdb"
    else:
        #Validation naming logic with timestamp
        base = os.path.splitext(os.path.basename(template_protein_structure_path))[0]
        timestamp = give_time_as_string()
        outdir = folder_to_save_validation_files or "."
        os.makedirs(outdir, exist_ok=True)
        name = f"{base}_{timestamp}"
        structure_path = os.path.join(outdir, f"{name}.pdb")
        generate_structure_from_sequence(protein_sequence, name=os.path.join(outdir, name))
    
    try:
        # 2) Compute helix/sheet %
        sse_arr = get_structural_annotations(structure_path)  # your existing function (chain A)
        helix_pct, sheet_pct = percentage_of_secondary_structure(
            sse_arr, 'both', starting_residue_id, ending_residue_id
        )

        # Optional: pLDDT gate if you still want it (not required for the array logic)
        if use_plddt:
            frac_conf = plddt_value_of_helical_residues(
                structure_path, starting_residue_id, ending_residue_id
            )
            # We can weave frac_conf into our final reward if desired.

        # 3) Reward based on helix vs sheet
        reward = get_reward_from_result(
            helix_pct, sheet_pct, cutoff_sheet=reward_cutoff_sheet
        )

        # 4) If sheet dominates, fill the 20-aa frequency array from neighborhood
        aa_counts = None
        if sheet_pct > helix_pct:
            aa_counts = count_aa_within_cutoff(
                structure_path,
                start_idx_0based=starting_residue_id,
                end_idx_0based=ending_residue_id,
                cutoff_angstrom=distance_cutoff,
                chain_id=chain_id,
                exclude_segment=True
            )

        # 5) Cleanup policy (match our previous behavior)
        if not validation:
            # Always remove in non-validation mode
            if os.path.exists(structure_path):
                os.remove(structure_path)
        else:
            # In validation mode, keep only if "bad" (we can invert if we want)
            if reward < 10 and os.path.exists(structure_path):
                os.remove(structure_path)

        return reward, aa_counts

    finally:
        # Ensure cleanup if something goes wrong and we're not validating
        if not validation and os.path.exists(structure_path):
            try:
                os.remove(structure_path)
            except Exception:
                pass
        


def reward_function(template_protein_structure_path,
                    protein_sequence,
                    reward_cutoff,
                    unique_name_to_give,
                    starting_residue_id,
                    ending_residue_id,
                    secondary_structure_type_from_env ='both',
                    validation=False,
                    folder_to_save_validation_files=None,
                    use_plddt = False
                    ):
    '''
    This function is used to calculate the reward based on the percentage of a specified secondary structure type in a segment of a protein.
    It generates the structure from the sequence using ESM model, annotates the secondary structure using biotite, and calculates the reward based on the criteria.
    This is called by the (gym) environment class to evaluate the rewards and is the most computationally expensive part
    Args:
        template_protein_structure_path (str): Path to the template protein structure file (used only in validation mode).
        protein_sequence (str): Amino acid sequence of the protein.
        reward_cutoff (float or tuple): Reward cutoff percentage.
        unique_name_to_give (str): Unique name to save the generated PDB file.
        starting_residue_id (int): Starting residue index (0-based) for the segment of interest.
        ending_residue_id (int): Ending residue index (0-based) for the segment of interest.
        secondary_structure_type_from_env (str): Type of secondary structure to calculate percentage for. Options are 'helix', 'sheet', or 'both'. Default is 'helix'.
        validation (bool): Whether to run in validation mode (saves files in a specified folder). Default is False.
        folder_to_save_validation_files (str): Folder path to save validation files (used only in validation mode).
        use_plddt (bool): Whether to consider pLDDT in the reward calculation. Default is False.
    Returns:
        float: Reward value based on the criteria.
    '''
    if validation==False:
        generate_structure_from_sequence(protein_sequence, name=f'NEW_{unique_name_to_give}')

        path_of_the_newly_created_file = f'NEW_{unique_name_to_give}.pdb'

        resultant_a_and_b_percentage = percentage_of_secondary_structure(get_structural_annotations(path_of_the_newly_created_file),
                                                   secondary_structure_type=secondary_structure_type_from_env,
                                                   starting_residue = starting_residue_id,
                                                   ending_residue = ending_residue_id) # here the starting and ending residues are also taken into account.
        
        result_from_plddt = plddt_value_of_helical_residues(structure_path = path_of_the_newly_created_file, 
                                                            starting_residue = starting_residue_id, 
                                                            ending_residue = ending_residue_id)
        reward = get_reward_from_result(result_pct_got=resultant_a_and_b_percentage,result_plddt=result_from_plddt,cutoff=reward_cutoff,usage_of_plddt=use_plddt)

        os.remove(path_of_the_newly_created_file)
        return reward

        
    if validation == True:
        
        template_file_base_name_without_extension = os.path.basename(template_protein_structure_path).split('.')[0]
        base_path_to_give_for_file = f'{folder_to_save_validation_files}/{template_file_base_name_without_extension}_{give_time_as_string()}'

        generate_structure_from_sequence(protein_sequence, name=base_path_to_give_for_file)
        path_of_the_newly_created_file = f'{base_path_to_give_for_file}.pdb'

        result = percentage_of_secondary_structure(get_structural_annotations(path_of_the_newly_created_file),
                                                   secondary_structure_type=secondary_structure_type_from_env,
                                                   starting_residue = starting_residue_id,
                                                   ending_residue = ending_residue_id) # here the starting and ending residues are also taken into account.
        
        result_from_plddt = plddt_value_of_helical_residues(structure_path = path_of_the_newly_created_file,
                                                            starting_residue = starting_residue_id,
                                                            ending_residue = ending_residue_id) # this is to obtain the fraction of residues that have a plddt >= 0.7 
        reward = get_reward_from_result(result_got=result,result_plddt=result_from_plddt,cutoff=reward_cutoff,usage_of_plddt=use_plddt)

        if int(reward)<10:
            os.remove(path_of_the_newly_created_file)
        return reward
