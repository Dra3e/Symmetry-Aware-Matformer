# run.py  – top‑level script
from training.ablation_studies.ablation_study import study   # just the import

# -----------------------------------------------------------------
if __name__ == "__main__":             # ← the “entry‑point guard”
    import multiprocessing as mp       # Windows needs this extra line
    mp.freeze_support()                # safe even when you’re *not* freezing an exe
    study()                            # everything that kicks off training