from model import PredictionModel
import pandas as pd
import os

root_path = 'Data/Data20230109_封存'

breath = pd.read_excel(os.path.join(root_path, 'breath.xlsx'), sheet_name='Sheet1')

# features = [f for f in breath]
# features.remove("category")
# features.remove("number")
# features.append('cancer_res')
# features.append('benign_res')

manners = {}
# features = ['m69.0335 ((C4H4O)H+) (Conc)', 'cancer_res']
features = ['m31.0178 ((CH2O)H+) (Conc)', 'm33.0335 ((CH4O)H+) (Conc)', 'm42.0338 ((C2H3N)H+) (Conc)', 'm49.0107 ((CH4S)H+) (Conc)', 'm59.0491 ((C3H6O)H+) (Conc)', 'm63.0263 ((C2H6S)H+) (Conc)', 'm68.0495 ((C4H5N)H+) (Conc)', 'm69.0335 ((C4H4O)H+) (Conc)', 'm69.0699 ((C5H8)H+) (Conc)', 'm77.0597 ((C3H8O2)H+) (Conc)', 'm85.0648 ((C5H8O)H+) (Conc)', 'm85.1012 ((C6H12)H+) (Conc)', 'm87.0441 ((C4H6O2)H+) (Conc)', 'm87.0804 ((C5H10O)H+) (Conc)', 'm89.0233 ((C3H4O3)H+) (Conc)', 'm89.0419 ((C4H8S)H+) (Conc)', 'm89.0597 ((C4H8O2)H+) (Conc)', 'm89.0961 ((C5H12O)H+) (Conc)', 'm95.0491 ((C6H6O)H+) (Conc)', 'm95.0855 ((C7H10)H+) (Conc)', 'm137.0709 ((C7H8N2O)H+) (Conc)', 'm137.1325 ((C10H16)H+) (Conc)'] # , 'cancer_res', 'benign_res']

# for feature in features:
#     manners[feature] = ['Standardization', 'Normalization']

path_train = os.path.join(root_path, 'train.xlsx')
path_eval = os.path.join(root_path, 'test.xlsx')

myModel = PredictionModel()
myModel.loadData(path_train, path_eval, features, "category")
myModel.dataProcessing(manners)
myModel.makeModel(model="lr", save_path="weight.pkl", mod="train")
myModel.makeModel(model="lr", pkl="weight.pkl", mod="val")
