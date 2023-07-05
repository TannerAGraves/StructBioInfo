# Classification of Contacts in Protein Structures

Structural Bioinformatics Project, Data Science @ UniPD, A.Y. 2022/23. Classification of Contacts in Protein Structures with Deep Neural Networks.

__Authors__: Tanner Graves, Marco Uderzo, Nour Alhousseini, Hazeezat Adebimpe Adebayo.

 <p align="center">
  <img width="500" alt="image" src="assets/protein_img.png">
</p>


### Project Context

Residue Interaction Networks are derived from protein structures based on geometrical and physico-chemical properties of the amino acids. RING is a software that takes a PDB file as input and returns the list of contacts (residue-residue pairs) and their types in a protein structure. RING contact types include:

- Hydrogen bonds (HBOND)
- Van der Waals interactions (VDW)
- Disulfide bridges (SBOND)
- Salt bridges (IONIC)
- π-π stacking (PIPISTACK) 
- π-cation (PICATION)
- Unclassified contacts

This project aims at predicting the RING classification of a contact based on supervised methods, rather than geometrical constraints. The software is able to calculate the propensity (or probability) of a contact belonging to each of the different contact types defined by RING, starting from the protein structure.


