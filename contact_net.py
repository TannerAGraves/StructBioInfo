import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for ContactNet")

    parser.add_argument("--in_pdb", type=str, required=False,
                        help="PDB ID.")

    parser.add_argument("--inference", action="store_true",
                        help="Flag to perform inference.")

    parser.add_argument("--train", action="store_true",
                        help="Flag to perform training.")

    return parser.parse_args()



def inference(): # this should load the model and perform inference given a pdb file
    pass



def train_model(): # this should retrain the model. To be determine if to allow hyperparameters tweaking.
    pass



def main():
    
    args = parse_arguments()

    input_pdb = args.in_pdb
    perform_inference = args.inference
    perform_training = args.train

    if perform_inference:
        if input_pdb == None:
            print("No PDB_id specified!")
            return
        print("Performing inference on PDB:", input_pdb)
        inference()
        
    elif perform_training:
        print("Performing re-training of ContactNet")
        train_model()
       
    else:
        print("No action specified. Use either --inference or --train.")







if __name__ == "__main__":
    main()




    