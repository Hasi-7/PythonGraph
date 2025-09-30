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

try:
    # Fit damped cosine model to data
    params, pcov = curve_fit(theta_func, t_data, theta_data, p0=initial_guess, maxfev=10000)
    fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
    perr = np.sqrt(np.diag(pcov))
    print(f"Best fit parameters for damped cosine:")
    print(f"  theta_0 = {fitted_theta_0:.4f} ± {perr[0]:.4f}")
    print(f"  tau     = {fitted_tau:.4f} ± {perr[1]:.4f} seconds")
    print(f"  T       = {fitted_T:.4f} ± {perr[2]:.4f} seconds")
    print(f"  phi_0   = {fitted_phi_0:.4f} ± {perr[3]:.4f} radians")
except RuntimeError as e:
    print("Damped cosine fit did not converge:", e)
    exit()

# Find peaks (local maxima) for envelope
peaks, _ = find_peaks(theta_data)
t_peaks = t_data[peaks]
theta_peaks = theta_data[peaks]

# Fit exponential decay to peaks for envelope
exp_guess = [np.max(theta_peaks), 10]  # Initial guess: [A, tau]
exp_params, exp_pcov = curve_fit(exp_decay, t_peaks, theta_peaks, p0=exp_guess)
A_fit, tau_fit = exp_params

# Print exponential envelope fit results
print(f"Exponential envelope fit:")
print(f"  A = {A_fit:.4f}")
print(f"  tau = {tau_fit:.4f} seconds")

# Plot data, damped cosine fit, and exponential envelope fit
plt.figure(figsize=(10, 6))
plt.scatter(t_data, theta_data, color='red', s=15, label='Measured Data')
t_fit = np.linspace(np.min(t_data), np.max(t_data), 1000)
theta_fit = theta_func(t_fit, *params)
plt.plot(t_fit, theta_fit, color='blue', label='Fitted Damped Cosine')
plt.plot(t_peaks, exp_decay(t_peaks, *exp_params), 'g--', label='Exponential Envelope Fit')
plt.scatter(t_peaks, theta_peaks, color='green', s=30, label='Envelope Peaks')
plt.xlabel('Time (s)')
plt.ylabel(r'Angle $\theta(t)$ [radians]')  # FIXED: single backslash
plt.title('Damped Pendulum Fit to $\theta(t) = \theta_0 e^{-t/\tau} \cos(2\pi t/T + \phi_0)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Q Factor:", len(peaks))