from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data.npy')
y_true = data[0]
y_pred = data[1]
y_proba = data[2]
proba_true, proba_pred = calibration_curve(y_true, y_proba, n_bins=5)
plt.rc('font', family='Times New Roman')
plt.plot(proba_pred, proba_true, 's-', label='Model Line')
plt.plot([0, 1], [0, 1], c='gray', ls='--', label='Perfect Line')
plt.title('Calibration Curve')
plt.xlabel('Mean Predicted Value')
plt.ylabel('Fraction of Positives')
plt.legend(loc=4)
plt.show()
print(proba_pred, proba_true)

coeff = np.polyfit(proba_pred, proba_true, 1)
print(coeff)
