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
X = df[['The thickness of PCM (mm)']]
y = df['Outpatient department']

# Ensure X is 2D
X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

# Create an object with secondary features
poly = PolynomialFeatures(degree=8)

# Transform features using this object
X_poly = poly.fit_transform(X)

# Create a model object
model = linear_model.LinearRegression()

# Fit the data
model.fit(X_poly, y)

# Print out the formula for polynomial regression
# Get the coefficients of the model
coefficients = model.coef_

# Get the polynomial term corresponding to each coefficient
powers = poly.powers_

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
formula += f"{model.intercept_}"
print(formula)

# Make predictions on the entire dataset
predictions = model.predict(X_poly)

plt.figure()

# Plot raw data points
plt.scatter(X, y, color='mediumpurple', label='Outpatient department')

# Draw the fitting curve
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, predictions), key=sort_axis)
X, predictions = zip(*sorted_zip)

x = np.linspace(0, 20, 400)
y1 = -3.439090e+00 * x + 1.807918e+00 * x**2 - 4.796184e-01 * x**3 + 7.089998e-02 * x**4 - 6.144393e-03 * x**5 + 3.101906e-04 * x**6 - 8.441244e-06 * x**7 + 9.565940e-08 * x**8 + 1.873567e+02
plt.plot(x, y1, "goldenrod",label="Fitted line y$_{1}$")

formula_text = r'$y_{1} = 187.3567 -3.439090 \cdot x + 1.807918 \cdot x^{2}$' + '\n' + \
r'$ -4.796184 \times 10^{-1} \cdot x^{3} + 7.089998 \times 10^{-2} \cdot x^{4}$' + '\n' + \
r'$ -6.144393 \times 10^{-3} \cdot x^{5} + 3.101906 \times 10^{-4} \cdot x^{6}$' + '\n' + \
r'$ -8.441244 \times 10^{-6} \cdot x^{7} + 9.565940 \times 10^{-8} \cdot x^{8}$' + '\n' + \
r''

plt.text(4, 184.9, formula_text, fontsize=12, fontdict={'fontname': 'Times New Roman', 'fontsize': 20})
from matplotlib.font_manager import FontProperties
font = FontProperties()
font.set_family('Times New Roman')

plt.xticks(np.arange(min(X), max(X)+1, 1.0))
plt.xlabel("The thickness of PCM (mm)")
plt.ylabel("Energy consumption (kW·h/m²·year)")
matplotlib.rcParams['font.sans-serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "sans-serif"

matplotlib.rcParams.update({'font.size': 15})
plt.legend()
plt.tight_layout()
plt.savefig('Outpatient department.png', dpi=300)
plt.show()

# Define function
x = sp.symbols('x')
f = -3.439090e+00 * x + 1.807918e+00 * x**2 - 4.796184e-01 * x**3 + 7.089998e-02 * x**4 - 6.144393e-03 * x**5 + 3.101906e-04 * x**6 - 8.441244e-06 * x**7 + 9.565940e-08 * x**8 + 1.873567e+02

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
annotation_text = f' ({max_slope_points[0]:.2f}, {max_slopes[0]:.2f})'
plt.annotate(annotation_text, (max_slope_points[0], max_slopes[0]), textcoords="offset points", xytext=(30,-45), ha='center', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=0, alpha=0.8))

plt.xlabel('The thickness of PCM (mm)')
plt.ylabel('Sensitivity of outpatient department')

plt.xticks(np.arange(0, 21, step=1))

plt.grid(True)
matplotlib.rcParams.update({'font.size': 15})

plt.legend(loc='best', facecolor='white', framealpha=1)

plt.tight_layout()
plt.savefig('Sensitivity of outpatient department.png', dpi=300)
plt.show()
