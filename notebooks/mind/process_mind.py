import numpy as np

def process_row(row):
    parts = row.strip().split('\t')

    id = parts[0]
    numbers = [float(num) for num in parts[1:]]

    return id, numbers

def get_embedding_dict(PATH):
    embedding_dict = {}
    with open(PATH, 'r') as f:
        lines = f.readlines()

    for line in lines:
        id, numbers = process_row(line)
        embedding_dict[id] = np.asarray(numbers)

    return embedding_dict

