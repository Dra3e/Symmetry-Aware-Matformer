from training.trainer import run

def main():
    print("Start of training on Gap_pbe!")
    run("gap pbe", "data/bulk_megnet_train.pkl", "data/bulk_megnet_val.pkl", 50)

if __name__ == "__main__":
    main()