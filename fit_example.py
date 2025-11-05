# Sample Python code to run the fit_black_box Python code relatively easily

from doctest import DocTestFailure
import fit_black_box as bb

# First, define the function you want to fit. Here it's a linear function.
# It is critical that the independant variable ("t") is first in the list of function variables.

def linear(t, m, b):
    return m*t + b

# Here's an exponential function where a is theta0 and b is tau

def expon(t, a, b):
    return a*t**(b)

# Next, generate your data and errorbars. One way is to manually insert it here.

# Note that xerr and yerr can either be an array of the same length as x&y, or a single value


# Now we make the plot, displayed on screen and saved in the directory, and print the best fit values

# Let's try again, this time loading from a file like a CSV file.
# NOTE: The CSV file should not have commas to separate things! Spaces or tabs are fine.

# Again, start with a fitting function. This time it is quadratic.

def quadratic(t, a, b, c):
    return a*(1 + b*t + c*(t**2))

def cubic(t, a, b, c, d):
    return a*t**3 + b*t**2 + c*t + d


# Now load the data from the file. The file should be in the same directory as this Python code.
# Some chance you will need an absolute path: "C:\\Users\\Brian\\Python\\mydata_fake.txt"

filename="mydata_fake.txt"
x, y, xerr, yerr = bb.load_data(filename)

# This time, let's use every single possible option available to bb.plot_fit()

init_guess = (-0.5, 0, +0.5) # guess for the best fit parameters
font_size = 20
xlabel = "Angle (Radians)"
ylabel = "Period (s)"

# Now we make the plot, displayed on screen and saved in the directory, and print the best fit values
bb.plot_fit(quadratic, x, y, xerr, yerr, init_guess=init_guess, font_size=font_size,
            xlabel=xlabel, ylabel=ylabel)

# Note: for sinusoidal functions, guessing the period correctly with init_guess is critical

# Fit the same data with an exponential function

# bb.plot_fit(expon, x, y, xerr, yerr)
# bb.plot_fit(cubic, x, y, xerr, yerr, init_guess=(1,1,1,1), font_size=font_size,
#             xlabel=xlabel, ylabel=ylabel)
# ============================
# NEW (revised): Invert quadratic for BOTH roots (±) with uncertainties
# ============================
import numpy as np
from scipy.optimize import curve_fit

# Refit using your loaded x, y, yerr and the same 'quadratic' model
def _refit_quadratic(x, y, yerr, p0):
    kwargs = {}
    try:
        if np.isscalar(yerr):
            sigma = np.full_like(y, float(yerr), dtype=float)
        else:
            sigma = np.asarray(yerr, dtype=float)
        if np.all(np.isfinite(sigma)) and np.all(sigma > 0):
            kwargs.update(sigma=sigma, absolute_sigma=True)
    except Exception:
        pass
    popt, pcov = curve_fit(quadratic, x, y, p0=p0, maxfev=200000, **kwargs)
    return popt, pcov

try:
    popt_q, pcov_q = _refit_quadratic(x, y, yerr, init_guess)
except Exception as e:
    print("Refit of quadratic failed:", e)
    raise

def _roots_for_y(a, b, c, y0):
    """
    Solve a*(1 + b t + c t^2) = y0  =>  (a c) t^2 + (a b) t + (a - y0) = 0.
    Returns *real* roots (sorted), possibly length 0, 1, or 2.
    """
    A = a * c
    B = a * b
    C = a - y0

    # Linear/degenerate case
    if abs(A) < 1e-15:
        if abs(B) < 1e-15:
            return np.array([])  # no information: constant
        return np.array([-C / B])  # single linear root

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return np.array([])  # complex roots
    sqrt_disc = np.sqrt(disc)
    r1 = (-B - sqrt_disc) / (2.0 * A)
    r2 = (-B + sqrt_disc) / (2.0 * A)
    out = np.array([r1, r2], dtype=float)
    out.sort()  # ascending: smaller (often negative) first
    return out

def _summarize(arr, name):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        print(f"{name}: no real samples.")
        return None
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan
    p16, p50, p84 = np.percentile(arr, [16, 50, 84])
    print(f"{name}: t = {mu:.6g} ± {sd:.2g}  (median {p50:.6g}; 16–84%: {p16:.6g}–{p84:.6g})")
    return mu, sd, (p16, p84)

# ---- Ask for target y, then compute nominal roots and MC uncertainties
try:
    y_target = float(input("\nEnter a target y value to invert the quadratic for t: ").strip())
except Exception:
    y_target = float(np.mean(y))
    print(f"No valid input detected; using y_target = {y_target:.6g}")

a0, b0, c0 = popt_q
nominal_roots = _roots_for_y(a0, b0, c0, y_target)
print("\nInverting y = a(1 + b t + c t^2) for your y_target, with parameter-uncertainty:")
print(f"Best-fit parameters (a, b, c) = {popt_q}")

if nominal_roots.size == 0:
    print("Nominal fit gives no real roots (discriminant < 0).")
elif nominal_roots.size == 1:
    print(f"Nominal root (linear/degenerate): t = {nominal_roots[0]:.6g}")
else:
    print(f"Nominal roots: t1 = {nominal_roots[0]:.6g} (smaller),  t2 = {nominal_roots[1]:.6g} (larger)")

# Monte Carlo: sample (a,b,c) from covariance, compute both roots each time
rng = np.random.default_rng(12345)
NSAMPLES = 20000
samples = rng.multivariate_normal(mean=popt_q, cov=pcov_q, size=NSAMPLES, check_valid="ignore")

r_small, r_large = [], []
for a_s, b_s, c_s in samples:
    rr = _roots_for_y(a_s, b_s, c_s, y_target)
    if rr.size == 2:
        r_small.append(rr[0])
        r_large.append(rr[1])
    elif rr.size == 1:
        # If only one root appears (near-degenerate), assign to the closer nominal root if available
        if nominal_roots.size == 2:
            if abs(rr[0] - nominal_roots[0]) <= abs(rr[0] - nominal_roots[1]):
                r_small.append(rr[0])
            else:
                r_large.append(rr[0])
        else:
            r_small.append(rr[0])  # fall back

print()  # spacing
_ = _summarize(r_small, "root1 (smaller t)")
_ = _summarize(r_large, "root2 (larger t)")