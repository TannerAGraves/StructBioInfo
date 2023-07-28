import argparse, logging, os
from Bio.PDB import PDBList
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model import ContactNet

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predict", action="store_true",
                        help="Perform prediction on new PDB. Default mode.")
    
    parser.add_argument("--pdb", type=str, required=False,
                    help="Specify input PDB code.")
    
    parser.add_argument("--full", action="store_true", required=False,
                        help="Generate output csv file with all features. DSSP is required.")
    
    parser.add_argument("--unprocessedfeats", action="store_true", required=False,
                    help="Generate output csv file with original, unprocessed model input features.")

    parser.add_argument("--train", action="store_true",
                        help="Perform training of the model.")
    


    return parser.parse_args()



def predict(model, input_pdb, full_out, unprocessedfeats):
    

    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(input_pdb, pdir="data/output/")
    logging.info("Calculating features...")
    os.system(f"python3 calc_features.py data/output/{input_pdb}.cif -out_dir data/output/") 

    try:
        df = pd.read_csv("data/output/" + input_pdb + ".tsv", sep='\t')
        logging.info(f"Successfully loaded {input_pdb}.tsv")
    except FileNotFoundError:
        logging.ERROR(f'File {input_pdb}.tsv does not exist, exiting.')
        exit()


    X = df[['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]
    X_orig=X.copy(deep=True)
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
    
    minMax = MinMaxScaler()
    minMax.fit(X)
    X_scaled = minMax.transform(X)

    logging.info("Predicting contacts...")
    y_pred = model.predict(X_scaled)

    X_scaled = pd.DataFrame(X_scaled)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(columns={0: "HBOND", 1: "IONIC", 2: "PICATION", 3: "PIPISTACK", 4: "SSBOND", 5: "VDW"})

    features = ['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
            't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']

    if full_out:
        out_data = pd.concat([df, y_pred], axis=1)
    elif unprocessedfeats:
        out_data = pd.concat([X_orig, y_pred], axis=1)
    else:
        out_data = pd.concat([X_scaled, y_pred], axis=1)
        length = len(features)
        for i in range(length):
            out_data = out_data.rename(columns={i: features[i]})



    #############################################################
    # Only if you want an additional Predicted Interaction Column

    #interaction_cols = out_data.iloc[:, -6:]
    #argmax_name = interaction_cols.idxmax(axis=1)
    #out_data['Predicted Interaction'] = argmax_name

    ############################################################

    out_data.to_csv(f"data/output/{input_pdb}_pred.csv")
    logging.info(f"Generated prediction file {input_pdb}_pred.csv in data/output/ folder.")





def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.basicConfig(format='[ContactNet]: %(message)s', level=logging.INFO)

    args = parse_arguments()

    input_pdb = args.pdb
    prediction_mode = args.predict
    training_mode = args.train
    full_out = args.full
    unprocessedfeats=args.unprocessedfeats


    contactnet = ContactNet()

    if prediction_mode:
        if input_pdb == None:
            logging.warning("No PDB_id specified!")
            return
        
        if full_out == True and unprocessedfeats == True:
            logging.warning("Cannot use --unprocessedfeats argument with --full argument. Please use either of the two.")
            return
        
        logging.info("Loading Pretrained model...")
        cn_model = contactnet.load_pretrained_model()
        logging.info(f"Processing PDB: {input_pdb}")
        predict(cn_model, input_pdb, full_out, unprocessedfeats)
        
    elif training_mode:
        logging.info("Performing Retraining of ContactNet")
        trained_model, hist = contactnet.train_model()
        logging.info("Saving model to /model folder.")
        contactnet.save_model(trained_model)
       
    else:
        logging.warning("No action specified. Use either --predict or --train. When using --predict, also specify --pdb your_pdb_id.")
        return






if __name__ == "__main__":
    main()




    