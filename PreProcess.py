import os
import pandas as pd
import json

root_path = "Data/Data20230327"

text = pd.read_excel(os.path.join(root_path, 'CTReportTextConcise.xlsx'), sheet_name="Sheet1")
category = pd.read_csv(os.path.join(root_path, 'CT list.csv'))

category["number"] = [str(i).split('\\')[0] for i in category["number"]]

file_train = open(os.path.join(root_path, 'train.txt'), 'w', encoding='utf-8')
file_test = open(os.path.join(root_path, 'test.txt'), 'w', encoding='utf-8')

number_cancer = 0
number_benign = 0

for idx, i in enumerate(text["ID"]):
    for jdx, j in enumerate(category["number"]):
        if str(i) == str(j):
            if category["category"][jdx] == 'cancer':
                number_cancer += 1
                elem = {'text': text["Text"][idx], 'label': category["category"][jdx]}
                if number_cancer % 4 != 0:
                    file_train.write(json.dumps(elem, ensure_ascii=False) + '\n')
                else:
                    file_test.write(json.dumps(elem, ensure_ascii=False) + '\n')
            else:
                number_benign += 1
                elem = {'text': text["Text"][idx], 'label': category["category"][jdx]}
                if number_benign % 4 != 0:
                    file_train.write(json.dumps(elem, ensure_ascii=False) + '\n')
                else:
                    file_test.write(json.dumps(elem, ensure_ascii=False) + '\n')

file_train.close()
file_test.close()

