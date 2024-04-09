import xgboost as xgb
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

# # model structure visualization
# model = xgb.Booster()
# model.load_model('weight.pkl')
# xgb.plot_tree(model, fmap='', num_trees=0, rankdir='UT', ax=None)
# plt.show()
#
# xgb.plot_importance(model)
# plt.show()

data = np.load('data.npy')

y_true = data[0]
y_pred = data[1]
y_proba = data[2]

y_idx = np.argsort(y_true)

y_true = y_true[y_idx]
y_pred = y_pred[y_idx]
y_proba = y_proba[y_idx]


x_b = np.linspace(1, 22, 22)
x_c = np.linspace(23, 57, 35)
plt.rc('font', family='Times New Roman')
plt.title('Prediction Probability Value')
plt.xlabel('Patient ID')
plt.ylabel('Probability')
plt.grid(color="k", linestyle=":")
plt.scatter(x_b, y_proba[0: 22], c='#32B897', marker='^', label='benign')
plt.scatter(x_c, y_proba[22:], c='#F27970', marker='v', label='cancer')
plt.axhline(0.5, c='red', ls='--')
plt.legend(loc=0)
plt.show()

# spectrum = pd.read_excel('SPECTRUM.xlsx', sheet_name='Sheet1')
# spectrum = spectrum.to_dict('list')
# spectrum.pop('Unnamed: 316')
# data = [[float(i.split(' ')[0][1:]), spectrum[i][0]] for i in spectrum]
# data = sorted(data, key=lambda x: x[0])
# data.insert(0, [17, 0])
# xy = []
# for i in range(len(data)-1):
#     xy.append(data[i])
#     num=1000
#     for k in range(num-1):
#         xy.append([(data[i+1][0]-data[i][0]) * (k+1)/ num + data[i][0], 0])
#
# print(data)
# print(xy)
#
# x = [i[0] for i in xy]
# y = [i[1] + 1 for i in xy]
#
# print(len(x), len(y))
# plt.rc('font', family='Times New Roman')
# plt.title('VOCs Spectrum')
# plt.grid(color="k", linestyle=":")
# # print(x[1295:])
# plt.semilogy(np.array(x), y, color='#2878b5', linestyle='-')
# plt.xlabel('Molecular mass (m/z)')
# plt.ylabel('VOCs concentration (ppb)')
# plt.axvline(59.0491, c='red')
# plt.axvline(101.0597)
# plt.show()

