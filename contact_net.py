import argparse, logging, os
from Bio.PDB import PDBList
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for ContactNet")

    parser.add_argument("--pdb", type=str, required=False,
                        help="PDB ID.")

    parser.add_argument("--inference", action="store_true",
                        help="Flag to perform inference.")

    parser.add_argument("--train", action="store_true",
                        help="Flag to perform training.")

    return parser.parse_args()



def inference(input_pdb): # this should load the model and perform inference given a pdb file
    #model = model()

    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(input_pdb, pdir=".")
    logging.info(f'Downloaded PDB: {input_pdb}')
    
    os.system(f"python calc_features.py {input_pdb}.cif") 

    df = pd.read_csv("./" + input_pdb + ".tsv", sep='\t')

    X = df[['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)

    print(X)


def train_model(): # this should retrain the model. To be determine if to allow hyperparameters tweaking.
    pass



def main():
    
    args = parse_arguments()

    input_pdb = args.pdb
    perform_inference = args.inference
    perform_training = args.train

    if perform_inference:
        if input_pdb == None:
            print("No PDB_id specified!")
            return
        print("Performing inference on PDB:", input_pdb)
        inference(input_pdb)
        
    elif perform_training:
        print("Performing re-training of ContactNet")
        train_model()
       
    else:
        print("No action specified. Use either --inference or --train.")







if __name__ == "__main__":
    main()




    