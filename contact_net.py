import argparse, logging, os
from Bio.PDB import PDBList
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model import ContactNet

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", type=str, required=False,
                        help="Specify input PDB code.")

    parser.add_argument("--inference", action="store_true",
                        help="Perform inference on new PDB.")

    parser.add_argument("--train", action="store_true",
                        help="Perform training of the model.")

    return parser.parse_args()



def perform_inference(model, input_pdb):
    

    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(input_pdb, pdir="/data/output/")
    logging.info(f'Downloaded PDB: {input_pdb}')
    
    os.system(f"python3 calc_features.py {input_pdb}.cif -out_dir data/output/") 

    df = pd.read_csv("data/output/" + input_pdb + ".tsv", sep='\t')

    X = df[['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
    
    minMax = MinMaxScaler()
    minMax.fit(X)
    X_scaled = minMax.transform(X)

    #print(X_scaled)


    y_pred = model.predict(X_scaled)

    X_scaled = pd.DataFrame(X_scaled)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(columns={0: "HBOND", 1: "IONIC", 2: "PICATION", 3: "PIPISTACK", 4: "SSBOND", 5: "VDW"})

    features = ['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
                't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']


    out_data = pd.concat([X, y_pred], axis=1)

    length = len(features)
    for i in range(length):
        out_data = out_data.rename(columns={i: features[i]})


    out_data.to_csv(f"{input_pdb}_contactnet_pred.csv") 





def main():

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

    args = parse_arguments()

    input_pdb = args.pdb
    inference_mode = args.inference
    training_mode = args.train


    contact_net_model = ContactNet(training_mode) # training_mode: bool


    if inference_mode:
        if input_pdb == None:
            logging.warning("No PDB_id specified!")
            return
        
        logging.info("Performing inference on PDB:", input_pdb)
        perform_inference(contact_net_model, input_pdb)
        
    elif training_mode:
        logging.info("Performing re-training of ContactNet")
        contact_net_model.train_model()
       
    else:
        logging.warning("No action specified. Use either --inference or --train.")
        return






if __name__ == "__main__":
    main()




    