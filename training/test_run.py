from training.dataset import extract_crystal_from_pkl
from training.trainer import run

def main():
    # print("Start of run!")
    # extract_crystal_from_pkl("data/bulk_megnet_train.pkl", "data/test_crystal.json", 1)
    print("Start of Training!")
    run("gap pbe", 1, "data/test_crystal.json")

if __name__ == "__main__":
    main()