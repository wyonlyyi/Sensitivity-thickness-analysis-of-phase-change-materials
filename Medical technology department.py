import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import operator
import matplotlib
import sympy as sp
from sympy import solveset, S

# Read Excel file
df = pd.read_excel('data.xlsx')

# Split the data into features and targets
X2 = df[['The thickness of PCM (mm)']]
y2 = df['Medical technology department']

# Ensure X is 2D
X2 = df.iloc[:, 0:1].values
y2 = df.iloc[:, 2].values

# Create an object with secondary features
poly2 = PolynomialFeatures(degree=6)

# Transform features using this object
X_poly2 = poly2.fit_transform(X2)

# Create a model object
model2 = linear_model.LinearRegression()

# Fit the data
model2.fit(X_poly2, y2)

# Print out the formula for polynomial regression
# Get the coefficients of the model
coefficients = model2.coef_
# Get the polynomial term corresponding to each coefficient
powers = poly2.powers_

# Construct the regression formula
formula = "y = "
for i, coef in enumerate(coefficients):
    if i == 0:
        continue
    term = []
    for j, power in enumerate(powers[i]):
        if power == 1:
            term.append(f"x_{j + 1}")
        elif power != 0:
            term.append(f"x_{j + 1}^{power}")
    if term:
        formula += f"{coef} * {' * '.join(term)} + "
formula += f"{model2.intercept_}"
print(formula)

# Make predictions on the entire dataset
predictions = model2.predict(X_poly2)
plt.figure()

# Plot raw data points
plt.scatter(X2, y2, color='royalblue', label='Medical technology department')

x = np.linspace(0, 20, 400)
y1 = -4.220646e-01 * x + 1.004508e-01 * x**2 - 1.265093e-02 * x**3 + 8.573089e-04 * x**4 - 2.987521e-05 * x**5 + 4.200435e-07 * x**6 + 2.719559e+02
plt.plot(x, y1, "lightseagreen",label="Fitted line y$_{2}$")

formula_text = r'$y_{2} = 271.9559 $' '\n' r'$ -4.220646 \times 10^{-1} \cdot x + 1.004508 \times 10^{-1} \cdot x^2$' '\n' r'$- 1.265093 \times 10^{-2} \cdot x^3 + 8.573089 \times 10^{-4} \cdot x^4$' '\n' r'$- 2.987521 \times 10^{-5} \cdot x^5 + 4.200435 \times 10^{-7} \cdot x^6$'

plt.text(4.5, 271.43, formula_text, fontsize=12)

plt.xticks(np.arange(min(X2), max(X2)+1, 1.0))
plt.xlabel("The thickness of PCM (mm)")
plt.ylabel("Energy consumption (kW·h/m²·year)")

matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

matplotlib.rcParams.update({'font.size': 15})

plt.legend()
plt.tight_layout()
plt.savefig('Medical technology department.png', dpi=300)

plt.show()

# Define function
x = sp.symbols('x')
f = -4.220646e-01 * x + 1.004508e-01 * x**2 - 1.265093e-02 * x**3 + 8.573089e-04 * x**4 - 2.987521e-05 * x**5 + 4.200435e-07 * x**6 + 2.719559e+02

# Compute derivative and second derivative
f_prime = f.diff(x)
f_double_prime = f.diff(x, 2)

# Create a function to calculate the sensitivity for a given x value
f_prime_lambdified = sp.lambdify(x, f_prime)

# Compute sensitivities for all x values
x_values = np.linspace(0, 20, 500)  # This can be tuned to the x-value range and precision you're interested in
sensitivities = f_prime_lambdified(x_values)

# Find the point with the largest derivative
max_slope_points = solveset(f_double_prime, x, domain=S.Reals)
max_slope_points = [point.evalf() for point in max_slope_points if 0 <= point <= 20]
max_slopes = [f_prime_lambdified(point) for point in max_slope_points]

# Create a graph to show the sensitivity
plt.figure()
plt.plot(x_values, sensitivities, label='Sensitivity')
plt.scatter(max_slope_points[0], max_slopes[0], color='goldenrod', s=100, label='Critical thickness for PCM')  # Only label the first point, set the size of the red point to 100

# Annotate the first maximum slope point
annotation_text = f'({max_slope_points[0]:.2f}, {max_slopes[0]:.2f})'
plt.annotate(annotation_text, (max_slope_points[0], max_slopes[0]), textcoords="offset points", xytext=(30,-30), ha='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=0, alpha=0.8))

plt.xlabel('The thickness of PCM (mm)')
plt.ylabel('Sensitivity of medical technology department')

plt.xticks(np.arange(0, 21, step=1))
matplotlib.rcParams.update({'font.size': 15})

plt.grid(True)
plt.legend(loc='best', facecolor='white', framealpha=1)  

plt.tight_layout()
plt.savefig('Sensitivity of medical technology department.png', dpi=300)
plt.show()
