import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Damped cosine function: theta(t) = theta_0 * exp(-t/tau) * cos(2*pi*t/T + phi_0)
def theta_func(t, theta_0, tau, T, phi_0):
    return theta_0 * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi_0)

# Exponential decay function for envelope: A * exp(-t/tau)
def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

# ----------------------------
# Optional: if you know your measurement noise (radians), set this to a float.
# When provided, curve_fit will use absolute_sigma=True, yielding absolute (not scaled) covariances.
# If unknown, leave as None to keep the default behavior.
known_sigma = None  # e.g., 0.002
# ----------------------------

# Load data (angle in radians in column 1, time in column 2)
data = np.loadtxt('QFactorGraph.txt', skiprows=1)
theta_data = data[:, 0]  # Angle (radians)
t_data = data[:, 1]      # Time (seconds)

# Print data for debugging
print('First 5 theta_data:', theta_data[:5])
print('First 5 t_data:', t_data[:5])

# Initial guess for damped cosine: [amplitude, tau, period, phase]
initial_guess = [theta_data[0], 10, 1, 0]
print('Initial guess for damped cosine:', initial_guess)

# Build kwargs for damped fit depending on whether we know sigma
fit_kwargs = dict(p0=initial_guess, maxfev=10000)
if known_sigma is not None:
    sigma = np.full_like(theta_data, known_sigma, dtype=float)
    fit_kwargs.update(sigma=sigma, absolute_sigma=True)

try:
    # Fit damped cosine model to data
    params, pcov = curve_fit(theta_func, t_data, theta_data, **fit_kwargs)
    fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
    perr = np.sqrt(np.diag(pcov))
    print("Best fit parameters for damped cosine (1σ):")
    print(f"  theta_0 = {fitted_theta_0:.6g} ± {perr[0]:.2g}")
    print(f"  tau     = {fitted_tau:.6g} ± {perr[1]:.2g} s")
    print(f"  T       = {fitted_T:.6g} ± {perr[2]:.2g} s")
    print(f"  phi_0   = {fitted_phi_0:.6g} ± {perr[3]:.2g} rad")
except RuntimeError as e:
    print("Damped cosine fit did not converge:", e)
    raise SystemExit

# ---- NEW: Propagate uncertainty to Q = π * tau / T using pcov from damped fit ----
# Gradient of Q with respect to [theta_0, tau, T, phi_0]
g = np.array([
    0.0,
    np.pi / fitted_T,
    -np.pi * fitted_tau / (fitted_T**2),
    0.0
], dtype=float)

Q = np.pi * fitted_tau / fitted_T
var_Q = float(g @ pcov @ g)
sigma_Q = np.sqrt(var_Q) if var_Q >= 0 else np.nan
print(f"Q = {Q:.6g} ± {sigma_Q:.2g}")

# Find peaks (local maxima) for envelope
peaks, _ = find_peaks(theta_data)
t_peaks = t_data[peaks]
theta_peaks = theta_data[peaks]

# Fit exponential decay to peaks for envelope
exp_guess = [np.max(theta_peaks), 10]  # Initial guess: [A, tau]
exp_params, exp_pcov = curve_fit(exp_decay, t_peaks, theta_peaks, p0=exp_guess, maxfev=10000)
A_fit, tau_fit = exp_params

# ---- NEW: 1σ uncertainties for envelope parameters ----
exp_perr = np.sqrt(np.diag(exp_pcov))
print("Exponential envelope fit (1σ):")
print(f"  A   = {A_fit:.6g} ± {exp_perr[0]:.2g}")
print(f"  tau = {tau_fit:.6g} ± {exp_perr[1]:.2g} s")

t_start, t_end = t_data[0], t_data[-1]

amp_start = fitted_theta_0 * np.exp(-t_start / fitted_tau)
amp_end   = fitted_theta_0 * np.exp(-t_end   / fitted_tau)

print(f"Amplitude at start (t={t_start:.2f} s): {amp_start:.6g} rad")
print(f"Amplitude at end   (t={t_end:.2f} s): {amp_end:.6g} rad")

# Plot data, damped cosine fit, and exponential envelope fit
plt.figure(figsize=(10, 6))
plt.scatter(t_data, theta_data, color='red', s=15, label='Measured Data')

t_fit = np.linspace(np.min(t_data), np.max(t_data), 1000)
theta_fit = theta_func(t_fit, *params)
plt.plot(t_fit, theta_fit, color='blue', label='Fitted Damped Cosine')

plt.plot(t_peaks, exp_decay(t_peaks, *exp_params), 'g--', label='Exponential Envelope Fit')
plt.scatter(t_peaks, theta_peaks, color='green', s=30, label='Envelope Peaks')

plt.xlabel('Time (s)')
plt.ylabel(r'Angle $\theta(t)$ [radians]')
plt.title('Pendulum Amplitude (Rad) Vs. Time (s) Graph')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()