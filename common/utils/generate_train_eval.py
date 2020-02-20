import random
import os


def generate_train_split(data_path, train_path, val_path, split_factor=0.1):
    dataset = []
    label2num = {}
    train_dir = os.path.basename(train_path)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    train_writer = open(train_path, 'w')
    val_dir = os.path.basename(val_path)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    val_writer = open(val_path, 'w')

    with open(data_path) as f:
        for line in f.readlines()[1:]:
            if random.random() < split_factor:
                val_writer.write(line)
            else:
                train_writer.write(line)
            idx, label = line.strip().split(',')
            if label not in label2num:
                label2num[label] = 0
            label2num[label] += 1
            dataset.append((idx, label))
    print('统计各类数据量为：')
    for label, n in label2num.items():
        print('%s类：%d' % (label, n))