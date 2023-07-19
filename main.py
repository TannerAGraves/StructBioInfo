import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for ContactNet")

    parser.add_argument("--input_pdb", type=str, required=True,
                        help="Filename of input PDB.")

    parser.add_argument("--inference", action="store_true",
                        help="Flag to perform inference.")

    parser.add_argument("--train", action="store_true",
                        help="Flag to perform training.")

    return parser.parse_args()



def inference(): # this should load the model and perform inference given a pdb file
    import os
    import pandas as pd
    
    if os.path.isdir(path):
        dfs = []
        for filename in os.listdir(path):
            if filename[-4:] == '.tsv':
                dfs.append(pd.read_csv(path + filename, sep='\t'))
                df = pd.concat(dfs)
    elif os.path.isfile(path):
        if filename[-4:] == '.tsv':
                df = pd.read_csv(path, sep='\t')
    else:
        print(f"Error: '{path}' does not exist on the system.")
    
    df = df[df.Interaction.notna()]
    contact_dict = {"HBOND": 0, "IONIC": 1, "PICATION": 2, "PIPISTACK": 3, "SSBOND": 4, "VDW": 5}
    y = df['Interaction']
    cat_names = list(y.astype('category').cat.categories)
    y.replace(contact_dict, inplace=True)
    '''TODO- apply the same feature subset from training'''
    X = df[['s_up', 's_down', 's_phi', 's_psi', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_up', 't_down', 't_phi', 't_psi', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']]
    X = X.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)
    '''TODO- check if the paramaters should be taking from training scaling to be applied here'''
    minMax = MinMaxScaler()
    minMax.fit(X)
    X_scaled = minMax.transform(X)

    pass



def train_model(): # this should retrain the model. To be determine if to allow hyperparameters tweaking.
    pass



def main():
    
    args = parse_arguments()

    input_pdb = args.input_pdb
    perform_inference = args.inference
    perform_training = args.train

    if perform_inference:
        print("Performing inference on PDB:", input_pdb)
        inference()
        
    elif perform_training:
        print("Performing re-training of ContactNet")
        train_model()
       
    else:
        print("No action specified. Use either --inference or --train.")







if __name__ == "__main__":
    main()




    