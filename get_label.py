label_set = set()
splits = ['train', 'test', 'dev']
for split in splits:
    f = open(f'data/{split}.sd.conllx')
    for line in f.readlines():
        if line != '\n':
            label = line.strip().split('\t')[-1]
            label_set.add(label)
            if label == '':
                print(line)

label_list = list(label_set)
label_list.sort()
print(label_list)