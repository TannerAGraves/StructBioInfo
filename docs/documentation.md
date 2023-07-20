# ContactNet Software Documentation

Authors: Tanner Graves, Marco Uderzo, Nour Alhousseini, Hazeezat Adebimpe Adebayo.

## Repository Overview and Files

The repository is organized as follows (only relevant files and folders are mentioned):

```
↳ data: folder containing all data.
        ↳ features_ring: folder containing all of the training data in .tsv format.
        ↳ output: folder containing the .tsv files outputted by calc-features.py

        🗎 atchley.tsv: utility file to compute features.
        🗎 ramachandran.dat: utility file to compute features.

↳ docs: folder containing the software documentation.
        🗎 documentation.md: software documentation.

↳ model: folder containing the trained model, loaded at inference time.
        🗎 model.keras: keras model file.
      
🗎 contact_net.py: main python script. run for inference or training, specifying arguments.
🗎 model.py: python script containing the untrained model.
🗎 calc-features.py: python script to compute the proteins features.
```

## Dependencies

Before running the software, make sure all the following dependent python libraries are installed on your machine:

- `Python 3.x`: assuming it's already installed.
- `biopython`
- `numpy`
- `pandas`
- `sklearn`
- `imblearn`
- `keras`
- `tensorflow`

To install all, run: `pip install biopython numpy pandas scikit-learn imbalanced-learn keras tensorflow` 

If you find python3 is not able to find the libraries, install them using the following command: `python3 -m pip install biopython numpy pandas scikit-learn imbalanced-learn keras tensorflow`



## Running the Software

The software allows:
- out-of-the-box inference by loading the trained model.
- complete retraining of the model.

### Inference Mode

- _Requires_: PDB code of target protein (referred below as `your_pdb_id`)
 
To run the software as default, run the following command in your terminal, using arguments:


`python3 contact_net.py --inference --pdb your_pdb_id`

TODO: Explain the .csv output 

### Training Mode

To train the model from scratch, run the following command:

`python3 contact_net.py --train`

Note that training is allowed with fixed hyperparameters. (Do we want to also allow full customization of hyperparams through arguments or a json config?)




