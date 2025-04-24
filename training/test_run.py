from training.dataset import extract_crystals_from_pkl
from training.trainer import run

def main():
    print("Start of run!")
    extract_crystals_from_pkl("data/bulk_megnet_train.pkl", 26)
    print("Start of Training!")
    run("gap pbe", "data/sample_crystals.pkl", "data/sample_crystals.pkl", 1)

if __name__ == "__main__":
    main()