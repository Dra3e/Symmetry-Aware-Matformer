from training.trainer import run

def main():
    """
    Main function to initiate the training process.

    This script starts the training process for predicting the "gap pbe" property
    using the specified training and validation datasets.
    """
    print("Start of training on Gap_pbe!")
    run("gap pbe", "data/bulk_megnet_train.pkl", "data/bulk_megnet_val.pkl", 50)

if __name__ == "__main__":
    main()