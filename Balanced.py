import os
import random

root_path = 'Data/Data20230327'

file = open(os.path.join(root_path, 'train.txt'), 'r', encoding='utf-8')
lines = file.readlines()
file.close()

cancer_lines = []
benign_lines = []
for line in lines:
    if 'cancer' in line:
        cancer_lines.append(line)
    else:
        benign_lines.append(line)

cancer_len = len(cancer_lines)
benign_len = len(benign_lines)
for i in range(cancer_len - benign_len):
    k = random.randint(0, len(benign_lines) - 1)
    benign_lines.append(benign_lines[k])

file_balanced = open(os.path.join(root_path, 'train_balanced.txt'), 'w', encoding='utf-8')
for i in range(len(cancer_lines)):
    file_balanced.write(cancer_lines[i])
    file_balanced.write(benign_lines[i])
file_balanced.close()
print(len(cancer_lines), len(benign_lines))
