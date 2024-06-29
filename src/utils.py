def preprocess_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    texts = []
    labels = []
    
    for line in lines:
        label, text = line.split(" ", maxsplit=1)
        texts.append(text)
        labels.append(int(label[-1]))
    return texts, labels