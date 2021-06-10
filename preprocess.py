import os
import random
import pickle

import numpy as np

from tqdm import tqdm

random.seed(0)
np.random.seed(0)

DATA_ROOT = "./drebin"
feature_vectors = os.path.join(DATA_ROOT, "feature_vectors")
sha256_family = os.path.join(DATA_ROOT, "sha256_family.csv")
ignore_types = ["url"]
apps = set(os.listdir(feature_vectors))
malwares = np.loadtxt(sha256_family, delimiter=",", skiprows=1, dtype=str)
malwares = set(malwares[:, 0].tolist())
benigns = apps.difference(malwares)
malwares = sorted(malwares)
benigns = sorted(benigns)

features = set()
for filename in tqdm(apps):
    with open(os.path.join(feature_vectors, filename), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip() 
            if line=="":
                continue
            feature_type = line.split("::")[0]
            if feature_type not in ignore_types:
                features.add(line)
features = sorted(features)
features = {v: k for k, v in enumerate(features)}

with open(os.path.join(DATA_ROOT, "features.pkl"), "wb") as f:
    pickle.dump(features, f)

