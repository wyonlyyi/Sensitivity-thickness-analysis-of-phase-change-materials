import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import operator
import sympy as sp
from sympy import Symbol, diff, solve

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

# Get the coefficients of the model
coefficients = model.coef_

# Define the polynomial
x = Symbol('x')
polynomial = (
    coefficients[8] * x**8
    + coefficients[7] * x**7
    + coefficients[6] * x**6
    + coefficients[5] * x**5
    + coefficients[4] * x**4
    + coefficients[3] * x**3
    + coefficients[2] * x**2
    + coefficients[1] * x
    + model.intercept_
)

print(polynomial)
# Compute the first and second derivatives
first_derivative = diff(polynomial, x)
second_derivative = diff(first_derivative, x)

# Define the functions for the polynomial and its derivatives
f = lambda x: polynomial.subs(Symbol('x'), x)
f_prime = lambda x: first_derivative.subs(Symbol('x'), x)
f_double_prime = lambda x: second_derivative.subs(Symbol('x'), x)

# Define the range for x
x_vals = np.linspace(0, 20, 400)
y_vals = [f(x) for x in x_vals]
y_prime_vals = [f_prime(x) for x in x_vals]
y_double_prime_vals = [f_double_prime(x) for x in x_vals]

# Find the inflection points by solving f''(x) = 0
inflection_points = solve(second_derivative, x)

# Plot the polynomial and its derivatives
plt.figure(figsize=(9, 12))

# Set global font to 'Times New Roman'
plt.rc('font', family='Times New Roman', size=15)

plt.subplot(3, 1, 1)
plt.plot(x_vals, y_vals, label='f (x)', color='forestgreen', linestyle='-')
plt.scatter(X, y, color='limegreen', marker='o', s=15, label='Simulated data')
plt.scatter([p for p in inflection_points if p.is_real], [f(p) for p in inflection_points if p.is_real], color='red', marker='x', s=60, label='Critical thickness of PCM')
offsets1 = [(0,15), (10,15), (10,15), (20,10), (10,12), (15,8)]
for idx, p in enumerate(inflection_points):
    if p.is_real:
        offset = offsets1[idx % len(offsets1)]
        plt.annotate(f'({p:.2f}, {f(p):.2f})', (p, f(p)), textcoords="offset points", xytext=offset, ha='center')
plt.ylabel('Energy consumption (kW·h/m²·year)')
plt.xlabel('The thickness of PCM (mm)')
plt.legend()
plt.grid(True, color='lightgrey')

#The coefficient marked in front of x comes from the coefficient displayed by print(polynomial) in the previous code
formula_text = r'$f (x) = 187.3567 -3.439090 \cdot x + 1.807918 \cdot x^{2}$' + '\n' + \
r'$ -4.796184 \times 10^{-1} \cdot x^{3} + 7.089998 \times 10^{-2} \cdot x^{4}$' + '\n' + \
r'$ -6.144393 \times 10^{-3} \cdot x^{5} + 3.101906 \times 10^{-4} \cdot x^{6}$' + '\n' + \
r'$ -8.441244 \times 10^{-6} \cdot x^{7} + 9.565940 \times 10^{-8} \cdot x^{8}$' + '\n' + \
r''
plt.text(0.65, 185.5, formula_text, fontsize=13)

plt.subplot(3, 1, 2)
plt.plot(x_vals, y_prime_vals, label="f '(x)", color='forestgreen', linestyle='--')
plt.scatter([p for p in inflection_points if p.is_real], [f_prime(p) for p in inflection_points if p.is_real], color='red', marker='x', s=60, label='Critical thickness of PCM')
plt.xlabel('The thickness of PCM (mm)')
plt.ylabel('First Derivative of f (x)')
plt.legend()
plt.grid(True, color='lightgrey')

plt.subplot(3, 1, 3)
plt.plot(x_vals, y_double_prime_vals, label="f ''(x)", color='forestgreen', linestyle='-.')
plt.scatter([p for p in inflection_points if p.is_real], [abs(f_double_prime(p)) for p in inflection_points if p.is_real], color='red', marker='x', s=60, label='Critical thickness of PCM')
offsets3 = [(10,10), (10,10), (0,10), (10,10), (0,10), (25,10)]
for idx, p in enumerate(inflection_points):
    if p.is_real:
        offset = offsets3[idx % len(offsets3)]
        plt.annotate(f'({p:.2f}, {abs(f_double_prime(p)):.2f})', (p, abs(f_double_prime(p))), textcoords="offset points", xytext=offset, ha='center')
plt.xlabel('The thickness of PCM (mm)')
plt.ylabel('Second Derivative of f (x)')
plt.legend()
plt.figtext(0, 1, '(a)', fontsize=20, fontweight='bold')
plt.grid(True, color='lightgrey')

plt.tight_layout()
plt.show()

inflection_points
