# ContactNet Software Documentation

Authors: Tanner Graves, Marco Uderzo, Nour Alhousseini, Hazeezat Adebimpe Adebayo.

## Repository Overview and Files

The repository is organized as follows (only relevant files and folders are mentioned):

```
↳ data: folder containing all data.
        ↳ features_ring: folder containing all of the training data in .tsv format.
        ↳ output: folder containing the .tsv files outputted by calc-features.py, as well as the output files of ContactNet.

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

It is assumed that `Python 3.x` is already installed.

Before running the software, make sure all the following dependent Python libraries are installed on your machine:

- `biopython`
- `numpy`
- `pandas`
- `sklearn`
- `imblearn`
- `keras`
- `tensorflow`

For visualization after training are also required:

- `matplotlib`
- `seaborn`

To install all, run: `pip install biopython numpy pandas scikit-learn imbalanced-learn keras tensorflow matplotlib seaborn` 

If you find python3 is not able to find the libraries, install them using the following command: `python3 -m pip install biopython numpy pandas scikit-learn imbalanced-learn keras tensorflow matplotlib seaborn`



## Running the Software

The software allows:
- out-of-the-box prediction by loading the pretrained model.
- complete retraining of the model.

### Inference Mode

- _Requires_: PDB code of target protein (referred below as `your_pdb_id`)
 
To run the software as default, run the following command in your terminal, using arguments:

`python3 contact_net.py --inference --pdb your_pdb_id`

*Usage Example*: run `python3 contact_net.py --inference --pdb 1a3n`.

Note that all generated files can be found in the `data/output/` folder. 

The software will download the pdb in .mmCIF format, compute its features using the `calc_features.py` script and import the corresponding generated `.tsv` file. In case the `.tsv` file is not found, the software will notify this and terminate its execution. Otherwise, the software will preprocess the data and run it through the pretrained model to predict the contacts.

Once finished, the DataFrame containing the protein's features will be merged with the one containing the probability distribution of each contact being of a certain type, giving more information about the confidence of the prediction.

The software will finally export the final `pdb_id_pred.csv` file in the `data/output/` folder. 

### Training Mode

To train the model from scratch, run the following command:

`python3 contact_net.py --train`




