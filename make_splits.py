# tools/make_splits.py
import json, random, pathlib
from sklearn.model_selection import train_test_split

RAW   = pathlib.Path("data/jarvis_splits/jdft_3d-12-12-2022.json")
DEST  = RAW.parent      # = data/jarvis_splits
DEST.mkdir(parents=True, exist_ok=True)

print("Loading full dataset …")
records = json.loads(RAW.read_text())
all_ids = [r["jid"] for r in records]

train, tmp = train_test_split(all_ids, test_size=0.20, random_state=42)
val,  test = train_test_split(tmp,     test_size=0.50, random_state=42)

for tag, ids in [("train",train),("val",val),("test",test)]:
    fn = DEST / f"dft_3d_{tag}_ids.json"
    fn.write_text(json.dumps(ids))
    print(f"wrote {fn}  ({len(ids)} ids)")