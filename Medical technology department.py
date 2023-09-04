import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import sympy as sp
from sympy import Symbol, diff, solve

# Read Excel file
df = pd.read_excel('data.xlsx')

# Split the data into features and targets
X = df[['The thickness of PCM (mm)']]
y = df['Medical technology department']

# Ensure X is 2D
X = df.iloc[:, 0:1].values
y = df.iloc[:, 2].values

# Create an object with secondary features
poly = PolynomialFeatures(degree=6)

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
    coefficients[6] * x**6
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

# Convert inflection_points to real numbers
inflection_points = [p.as_real_imag()[0] for p in inflection_points]

# Plot the polynomial and its derivatives
plt.figure(figsize=(9, 12))

# Set global font to 'Times New Roman'
plt.rc('font', family='Times New Roman', size=15)

plt.subplot(3, 1, 1)
plt.plot(x_vals, y_vals, label='f (x)', color='goldenrod', linestyle='-')
plt.scatter(X, y, color='orange', marker='o', s=15, label='Simulated data')
plt.scatter([p for p in inflection_points if p.is_real], [f(p) for p in inflection_points if p.is_real], color='blue', marker='x', s=60, label='Critical thickness of PCM')
offsets1 = [(0,20), (-10,20), (10,12), (20,15)]
for idx, p in enumerate(inflection_points):
    if p.is_real:
        offset = offsets1[idx % len(offsets1)]
        plt.annotate(f'({p:.2f}, {f(p):.2f})', (p, f(p)), textcoords="offset points", xytext=offset, ha='center')
plt.ylabel('Energy consumption (kW·h/m²·year)')
plt.xlabel('The thickness of PCM (mm)')
plt.legend()
plt.grid(True, color='lightgrey')

#The coefficient marked in front of x comes from the coefficient displayed by print(polynomial) in the previous code
formula_text = r'$f (x) = 271.9559 $' '\n' r'$ -4.220646 \times 10^{-1} \cdot x + 1.004508 \times 10^{-1} \cdot x^2$' '\n' r'$- 1.265093 \times 10^{-2} \cdot x^3 + 8.573089 \times 10^{-4} \cdot x^4$' '\n' r'$- 2.987521 \times 10^{-5} \cdot x^5 + 4.200435 \times 10^{-7} \cdot x^6$'
plt.text(0.75, 271.67, formula_text, fontsize=13)

plt.subplot(3, 1, 2)
plt.plot(x_vals, y_prime_vals, label="f '(x)", color='goldenrod', linestyle='--')
plt.scatter([p for p in inflection_points if p.is_real], [f_prime(p) for p in inflection_points if p.is_real], color='blue', marker='x', s=60, label='Critical thickness of PCM')
plt.xlabel('The thickness of PCM (mm)')
plt.ylabel('First Derivative of f (x)')
plt.legend()
plt.grid(True, color='lightgrey')

plt.subplot(3, 1, 3)
plt.plot(x_vals, y_double_prime_vals, label="f ''(x)", color='goldenrod', linestyle='-.')
plt.scatter([p for p in inflection_points if p.is_real], [abs(f_double_prime(p)) for p in inflection_points if p.is_real], color='blue', marker='x', s=60, label='Critical thickness of PCM')
offsets = [(0,10), (-35,10), (0,10), (-10,10)]  # You can define more offsets if you have more points
for idx, p in enumerate(inflection_points):
    if p.is_real:
        offset = offsets[idx % len(offsets)]  # Cycle through the list of offsets
        plt.annotate(f'({p:.2f}, {abs(f_double_prime(p)):.2f})', (p, abs(f_double_prime(p))), textcoords="offset points", xytext=offset, ha='center')
plt.xlabel('The thickness of PCM (mm)')
plt.ylabel('Second Derivative of f (x)')
plt.legend()

plt.figtext(0, 1, '(b)', fontsize=20, fontweight='bold')

plt.grid(True, color='lightgrey')

plt.tight_layout()
plt.show()

inflection_points