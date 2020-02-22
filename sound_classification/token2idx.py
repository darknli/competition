def convert2idx(label1, label2):
    if label1 == 'A':
        return int(label2)
    elif label1 == 'B':
        return int(label2) + 10
    elif label1 == 'C':
        return int(label2) + 20
    elif label1 == 'D':
        return int(label2) + 30
    elif label1 == 'E':
        return int(label2) + 40
    else:
        print(label1)