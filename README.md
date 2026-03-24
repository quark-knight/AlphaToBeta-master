 # AlphaToBeta: A deep reinforcement learning model for predicting amino-acid mutations that induce α-helix → β-sheet transitions in proteins
 This is the code repository for the AlphaToBeta Project at Prof. Arnab Mukherjee's CCB lab at IISER Pune.

 **The code is still being changed, and the model is in the training phase**

The question we want to answer is whether we have a predictive model that can predict point amino acid mutations, which lead to α-helix → β-sheet transitions in proteins.

We are using ESMFold with a joint-embedding space of both the helix sequence and environmental sequence with the objective to maximize the reward, which basically mean getting heigher α-helix than β-sheet in secondary structure determination. 

`Check AlphaToBeta_synced` Folder for updated versions that have code changes done at the computer Cluster.

![Picture1](https://github.com/user-attachments/assets/e6e484f6-ab3b-410e-b56e-70cd47166df9)
