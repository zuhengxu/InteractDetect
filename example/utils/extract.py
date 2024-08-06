import torch
import numpy as np

def extract_feature(df, words):
    X = np.array(df.filter(like=words))
    return torch.tensor(X, dtype=torch.float32)


def combine_dicts(dic1, dic2):
    # always combine dic2 into dic1
    if not dic1: 
        return dic2

    result = {}
    if not dic2: 
        raise ValueError("dic2 is empty")
    for key in dic2.keys():
        if isinstance(dic1[key], list):
            result[key] = dic1[key]
            result[key].append(dic2[key])
        else:
            if key in dic1: result.setdefault(key, []).append(dic1[key])
            if key in dic2: result.setdefault(key, []).append(dic2[key])
    return result

# dic1 = {'A': 25, 'B': 41, 'C': 32}
# dic2 = {'A': 21, 'B': 12, 'C': 62}
# c = combine_dicts({}, dic2)




