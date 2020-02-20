import random

def split_data(labels, boxes, train_path, val_path, split_factor=0.2):

    box_map = {}
    with open(boxes) as f:
        boxes = f.readlines()[1:]
        for box in boxes:
            fields = box.strip().split(',')
            img, box = fields[0], fields[1:]
            box_map[img] = box

    train_writer = open(train_path, 'w')
    val_writer = open(val_path, 'w')
    with open(labels) as f:
        for line in f.readlines()[1:]:
            fields = line.strip().split(',')
            img, label = fields
            if label != '0':
                box = box_map[img]
            else:
                box = ['0', '0', '0', '0']
            fields.extend(box)
            line = ','.join(fields) + '\n'
            if random.random() > split_factor:
                train_writer.write(line)
            else:
                val_writer.write(line)
    train_writer.close()
    val_writer.close()

