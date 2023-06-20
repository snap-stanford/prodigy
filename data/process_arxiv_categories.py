import pandas as pd
import os

rel_path = "./arxiv_CS_categories.txt"


f = open(os.path.join(os.path.dirname(__file__), rel_path), "r").readlines()

state = 0
result = {
    "id": [],
    "name": [],
    "description": []
}

for line in f:
    if state == 0:
        assert line.strip().startswith("cs.")
        category = "arxiv " + " ".join(line.strip().split(" ")[0].split(".")).lower() # e. g. cs lo
        name = line.strip()[7:-1]  # e. g. Logic in CS
        result["id"].append(category)
        result["name"].append(name)
        state = 1
        continue
    elif state == 1:
        description = line.strip()
        result["description"].append(description)
        state = 2
        continue
    elif state == 2:
        state = 0
        continue

arxiv_cs_taxonomy = pd.DataFrame(result)