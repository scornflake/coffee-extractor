import numpy as np
from scipy.optimize import curve_fit

# Your data
x = np.array([
14.99874725,
29.99749451,
44.99624176,
59.99498901,
74.99373627,
104.9912308,
329.9724396,
419.9649231,
449.9624176,
464.9611649,
494.9586594,
509.9574066,
524.9561539,
554.9536484,
749.9373627,
764.9361099,
794.9336044,
809.9323517,
824.9310989,
869.9273407,
884.926088,
914.9235825,
959.9198242,
1004.916066,
1019.914813,
1049.912308,
])

y = np.array([
38,
62,
81,
95,
106,
122,
182,
195,
198,
200,
202,
203,
204,
209,
222,
223,
225,
226,
228,
229,
230,
231,
232,
233,
234,
236
])

# Define the form of the function we want to fit (in this case, a polynomial of degree 2).
def func(x, a, b, c):
    return a * x**2 + b * x + c

# Use curve_fit to find the best parameters a, b, c
popt, pcov = curve_fit(func, x, y)

print(f"The equation of best fit is: y = {popt[0]}*x^2 + {popt[1]}*x + {popt[2]}")
