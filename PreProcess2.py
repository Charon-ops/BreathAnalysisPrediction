import os
import pandas as pd
import json

root_path = "Data/Data20221122"

text = pd.read_excel(os.path.join(root_path, 'CTReportTextConcise.xlsx'), sheet_name="Sheet1")
category = pd.read_excel(os.path.join(root_path, 'breath.xlsx'), sheet_name='Sheet1')

voc = ["m137.0709 ((C7H8N2O)H+) (Conc)", "m31.0178 ((CH2O)H+) (Conc)", "m42.0338 ((C2H3N)H+) (Conc)",
       "m49.0107 ((CH4S)H+) (Conc)", "m59.0491 ((C3H6O)H+) (Conc)", "m63.0263 ((C2H6S)H+) (Conc)",
       "m69.0699 ((C5H8)H+) (Conc)", "m77.0597 ((C3H8O2)H+) (Conc)", "m85.1012 ((C6H12)H+) (Conc)",
       "m87.0441 ((C4H6O2)H+) (Conc)", "m89.0233 ((C3H4O3)H+) (Conc)", "m89.0961 ((C5H12O)H+) (Conc)",
       "m95.0491 ((C6H6O)H+) (Conc)"]

category["number"] = [i.split('\\')[0] for i in category["number"]]

file_train = open('Data/Data20221122/train2.txt', 'w', encoding='utf-8')
file_test = open('Data/Data20221122/test2.txt', 'w', encoding='utf-8')

number_cancer = 0
number_benign = 0

for idx, i in enumerate(text["ID"]):
    for jdx, j in enumerate(category["number"]):
        if str(i) == str(j):
            if category["category"][jdx] == 'cancer':
                number_cancer += 1
                content = text["Text"][idx]
                for f in voc:
                    content = content + ',' + str(category[f][jdx])
                elem = {'text': content, 'label': category["category"][jdx]}
                if number_cancer % 4 != 0:
                    file_train.write(json.dumps(elem, ensure_ascii=False) + '\n')
                else:
                    file_test.write(json.dumps(elem, ensure_ascii=False) + '\n')
            else:
                number_benign += 1
                content = text["Text"][idx]
                for f in voc:
                    content = content + ',' + str(category[f][jdx])
                elem = {'text': content, 'label': category["category"][jdx]}
                if number_cancer % 4 != 0:
                    file_train.write(json.dumps(elem, ensure_ascii=False) + '\n')
                else:
                    file_test.write(json.dumps(elem, ensure_ascii=False) + '\n')

file_train.close()
file_test.close()
