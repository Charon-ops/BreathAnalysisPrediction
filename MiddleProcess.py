from copy import copy

import pandas as pd
import os

from tqdm import tqdm

root_path = 'Data/Data20230109'

text = pd.read_excel(os.path.join(root_path, 'CTReportTextConcise.xlsx'), sheet_name='Sheet1')
breath = pd.read_excel(os.path.join(root_path, 'breath.xlsx'), sheet_name='Sheet1')
category = pd.read_csv(os.path.join(root_path, 'CT list.csv'))

train_nlp = pd.read_excel(os.path.join(root_path, 'train_nlp.xlsx'), sheet_name='Sheet1')
test_nlp = pd.read_excel(os.path.join(root_path, 'test_nlp.xlsx'), sheet_name='Sheet1')

vocs = [f for f in breath]
vocs.remove('number')
vocs.remove('category')

train_data = {'id': [], 'cancer_res': [], 'benign_res': [], 'value_res': [], 'category': []}
for f in vocs:
    train_data[f] = []


for jdx, j in enumerate(train_nlp['text']):
    idx = list(text['Text']).index(j)
    ID = str(text['ID'][idx]) + '\\'
    if ID not in [i.split('\\')[0] + "\\" for i in list(breath["number"])] or ID not in [i.split('\\')[0] + "\\" for i in list(category["number"])]:
        print(ID)
        continue
    kdx = list([i.split('\\')[0] + "\\" for i in list(breath["number"])]).index(ID)
    zdx = list([i.split('\\')[0] + "\\" for i in list(category["number"])]).index(ID)

    train_data['id'].append(ID)
    train_data['cancer_res'].append(train_nlp['report_cancer'][jdx])
    train_data['benign_res'].append(train_nlp['report_benign'][jdx])
    train_data['value_res'].append(train_nlp['report_value'][jdx])
    train_data['category'].append(1 if category['category'][zdx] == 'cancer' else 0)
    for f in vocs:
        train_data[f].append(breath[f][kdx])

test_data = {'id': [], 'cancer_res': [], 'benign_res': [], 'value_res': [], 'category': []}
for f in vocs:
    test_data[f] = []

for jdx, j in enumerate(test_nlp['text']):
    idx = list(text['Text']).index(j)
    ID = str(text['ID'][idx]) + '\\'
    if ID not in [i.split('\\')[0] + "\\" for i in list(breath["number"])] or ID not in [i.split('\\')[0] + "\\" for i in list(category["number"])]:
        print(ID)
        continue
    kdx = list([i.split('\\')[0] + "\\" for i in list(breath["number"])]).index(ID)
    zdx = list([i.split('\\')[0] + "\\" for i in list(category["number"])]).index(ID)

    test_data['id'].append(ID)
    test_data['cancer_res'].append(test_nlp['report_cancer'][jdx])
    test_data['benign_res'].append(test_nlp['report_benign'][jdx])
    test_data['value_res'].append(test_nlp['report_value'][jdx])
    test_data['category'].append(1 if category['category'][zdx] == 'cancer' else 0)
    for f in vocs:
        test_data[f].append(breath[f][kdx])

train_data = pd.DataFrame(train_data)
train_data.to_excel(os.path.join(root_path, 'train.xlsx'), index=False)
test_data = pd.DataFrame(test_data)
test_data.to_excel(os.path.join(root_path, 'test.xlsx'), index=False)
# print(a)



