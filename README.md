
<h1 align="center"> Classification of Contacts in Protein Structures</h1>


<p align="center">
<b>Structural Bioinformatics Project, Data Science @ UniPD, A.Y. 2022/23. </b>
</p>

<p align="center">
<b> Authors: </b>Tanner Graves, Marco Uderzo, Nour Alhousseini, Hazeezat Adebimpe Adebayo.
</p>

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

### Repository Overview and Files

The repository is organized as follows (only relevant files and folders mentioned):

```
↳ bin/dssp: contains mkdssp. Recompiling might be necessary.

↳ data: folder containing all data.
        ↳ features_ring: folder containing all of the training data in .tsv format.
                        🗎 *.tsv
        ↳ output: output folder containing all output files of the software.

        🗎 atchley.tsv: utility file to compute features.
        🗎 ramachandran.dat: utility file to compute features.

↳ docs: folder containing the software documentation.
        🗎 documentation.md: software documentation.

↳ model: folder containing the trained model, loaded at prediction time.
        🗎 model.keras: keras model file.
      
🗎 contact_net.py: main python script. run for inference or training, specifying arguments.
🗎 model.py: python script containing the untrained model.
🗎 calc-features.py: python script to compute the proteins features.
🗎 configuration.json: JSON file containing configuration settings.
🗎 contacts_classification_keras.ipynb: Jupyter notebook of ContactNet.
🗎 project_report.pdf: final project report for the Structural Bioinformatics course.
```

