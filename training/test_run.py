from training.dataset import extract_crystal_from_pkl, convert_pkl_to_json
from training.trainer import run

def main():
    # print("Start of run!")
    # # convert_pkl_to_json("data/bulk_megnet_train.pkl", "data/bulk_megnet_train.json")
    # extract_crystal_from_pkl("data/bulk_megnet_train.pkl", "data/test_crystal.json", 1)
    print("Start of Training!")
    run("gap pbe", "data/test_crystal.json", 100)

if __name__ == "__main__":
    main()