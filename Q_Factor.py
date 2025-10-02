import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ----------------------------
# Model functions
# ----------------------------
def theta_func(t, theta_0, tau, T, phi_0):
    """Damped cosine: θ(t) = θ0 exp(-t/τ) cos(2πt/T + φ0)"""
    return theta_0 * np.exp(-t / tau) * np.cos(2 * np.pi * t / T + phi_0)

def exp_decay(t, A, tau):
    """Exponential envelope: A exp(-t/τ)"""
    return A * np.exp(-t / tau)

# ----------------------------
# Load data (angle [rad], time [s])
# ----------------------------
data = np.loadtxt('QFactorGraph2.txt', skiprows=1)
theta_data = data[:, 0].astype(float)
t_data     = data[:, 1].astype(float)

print("First 5 theta_data:", theta_data[:5])
print("First 5 t_data:", t_data[:5])

# ----------------------------
# Measurement uncertainties
# ----------------------------
time_sigma  = 1.0/30.0   # one timestamp uncertainty: 1 frame at 30 fps
angle_sigma = 0.002      # ~0.1° ≈ 0.002 rad

# ----------------------------
# Damped cosine fit (for overlay/residuals only)
# ----------------------------
initial_guess = [theta_data[0], 200.0, 1.5, 0.0]
params, pcov = curve_fit(theta_func, t_data, theta_data, p0=initial_guess, maxfev=20000)
fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
perr = np.sqrt(np.diag(pcov))

print("\nDamped-cosine fit (±1σ):")
print(f"  theta0 = {fitted_theta_0:.6g} ± {perr[0]:.2g} rad")
print(f"  tau    = {fitted_tau:.6g} ± {perr[1]:.2g} s")
print(f"  T      = {fitted_T:.6g} ± {perr[2]:.2g} s")
print(f"  phi0   = {fitted_phi_0:.6g} ± {perr[3]:.2g} rad")

# ----------------------------
# Maxima detection (envelope points)
# ----------------------------
peaks, _ = find_peaks(theta_data, prominence=0.001)
if peaks.size < 2:
    raise SystemExit("Not enough peaks found to compute periods.")

t_peaks     = t_data[peaks]
theta_peaks = theta_data[peaks]
A_peaks     = np.abs(theta_peaks)

# ----------------------------
# Envelope fit on maxima
# ----------------------------
exp_guess  = [np.max(A_peaks), max(1.0, fitted_tau/2)]
exp_params, exp_pcov = curve_fit(exp_decay, t_peaks, A_peaks, p0=exp_guess, maxfev=20000)
A_fit, tau_fit = exp_params
exp_perr = np.sqrt(np.diag(exp_pcov))
sigma_tau = exp_perr[1]

print("\nExponential envelope fit (±1σ):")
print(f"  A   = {A_fit:.6g} ± {exp_perr[0]:.2g} rad")
print(f"  tau = {tau_fit:.6g} ± {sigma_tau:.2g} s")

# ----------------------------
# 20% points (STRICT ±1% band), maxima only
# ----------------------------
A0     = A_peaks[0]
target = 0.20 * A0
tol    = 0.01 * target        # ±1% of the 20% target
near20_mask = (A_peaks >= target - tol) & (A_peaks <= target + tol)

idx20 = np.where(near20_mask)[0]
if idx20.size > 0:
    first_20_idx = int(idx20[0])
    N_count           = first_20_idx + 1      # oscillations to FIRST ~20% peak
    sigma_N_count     = 1                     # ±1 oscillation ambiguity
    Q_count           = 2 * N_count
    sigma_Q_count     = 2                     # ±2 from ±1 oscillation
else:
    first_20_idx = None
    N_count = sigma_N_count = Q_count = sigma_Q_count = None

# Arrays for orange markers (maxima plot only)
t_20 = t_peaks[near20_mask]
A_20 = A_peaks[near20_mask]

print(f"\n20% target amplitude = {target:.6f} rad, allowed band = [{target - tol:.6f}, {target + tol:.6f}] rad")
print(f"Number of maxima within ±1% of target: {t_20.size}")
if first_20_idx is not None:
    print(f"N_count (to first 20%) = {N_count} ± {sigma_N_count}")
    print(f"Q_count = 2·N_count = {Q_count} ± {sigma_Q_count}")
else:
    print("No maxima within ±1% of the 20% target were found.")

# ----------------------------
# Mean period from total elapsed time and its uncertainty
# ----------------------------
# Use ONLY the first and last detected peaks to define N periods robustly
N_periods = t_peaks.size - 1
t_total   = t_peaks[-1] - t_peaks[0]
T_bar     = t_total / N_periods

# Uncertainty of T_bar from timestamp uncertainty:
# t_total = t_last - t_first, so σ_t_total = sqrt(σ_t^2 + σ_t^2) = sqrt(2) σ_t
sigma_T_bar = (np.sqrt(2.0) * time_sigma) / N_periods

print(f"\nMean period T̄ = {T_bar:.6g} ± {sigma_T_bar:.2g} s  (from N={N_periods} periods)")

# ----------------------------
# Q2 and its propagated uncertainty
# ----------------------------
Q2 = np.pi * tau_fit / T_bar
# dQ/dtau = π / T_bar ; dQ/dT = -π τ / T_bar^2
dQ_dtau = np.pi / T_bar
dQ_dT   = -np.pi * tau_fit / (T_bar**2)
sigma_Q2 = np.sqrt((dQ_dtau * sigma_tau)**2 + (dQ_dT * sigma_T_bar)**2)

print(f"Q2 = {Q2:.6g} ± {sigma_Q2:.2g}")

# ----------------------------
# Plot 1: Main time series + cosine fit (NO orange markers here)
# ----------------------------
plt.figure(figsize=(12, 6))
plt.errorbar(
    t_data, theta_data,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', markersize=2.2, color='tab:red', alpha=0.55,
    ecolor='lightgray', elinewidth=0.8, capsize=0,
    label='Measured data (±σ)', zorder=1
)

t_plot = np.linspace(t_data.min(), t_data.max(), 2000)

# Smooth envelope curve (for context)
plt.plot(
    t_plot, exp_decay(t_plot, A_fit, tau_fit),
    linestyle='--', linewidth=2.5, color='magenta',
    label='Exponential envelope fit', zorder=3
)
# Peaks with error bars (green)
plt.errorbar(
    t_peaks, theta_peaks,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', markersize=4, color='tab:green',
    ecolor='lightgray', capsize=0,
    label='Envelope peaks', zorder=2
)

# Cosine fit LAST so it overlays on top
plt.plot(
    t_plot, theta_func(t_plot, *params),
    color='tab:blue', linewidth=2.2, label='Damped-cosine fit', zorder=4
)

plt.xlabel('Time (s)')
plt.ylabel(r'Angle $\theta(t)$ [rad]')
plt.title('Damped Pendulum: Data w/ Error Bars, Cosine Fit (on top), and Envelope')
plt.grid(True, alpha=0.25)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 2: Maxima-only — envelope fit + residuals (+ orange markers ON TOP)
# ----------------------------
A_fit_at_peaks = exp_decay(t_peaks, A_fit, tau_fit)
env_residuals  = A_peaks - A_fit_at_peaks

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True,
    gridspec_kw={'height_ratios': [3, 2]}
)

# Top: maxima with error bars (blue)
ax1.errorbar(
    t_peaks, A_peaks,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', ms=4, color='tab:blue', ecolor='lightgray',
    alpha=0.85, label='Maxima (±σ)', zorder=2
)
# Envelope line
ax1.plot(t_peaks, A_fit_at_peaks, 'g--', lw=2.2, label='Exponential envelope fit', zorder=3)

# ORANGE points plotted LAST with high zorder to ensure they sit on top
if t_20.size > 0:
    ax1.scatter(
        t_20, A_20,
        s=70, facecolor='orange', edgecolor='black', linewidth=0.9,
        label='20% amplitude peaks (±1%)', zorder=5
    )

ax1.set_ylabel('Peak amplitude [rad]')
ax1.set_title('Exponential Envelope Fit on Maxima (20% peaks highlighted)')
ax1.grid(True, alpha=0.25)
ax1.legend(loc='upper right', frameon=True)

# Bottom: residuals with error bars
ax2.axhline(0, color='k', lw=1)
ax2.errorbar(
    t_peaks, env_residuals,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', ms=3, ecolor='lightgray', elinewidth=1,
    capsize=0, color='black', label='Residuals (data - fit)'
)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Residual [rad]')
ax2.set_title('Residuals of Exponential Fit')
ax2.grid(True, axis='y', linestyle=':')
ax2.legend(loc='upper right', frameon=True)

fig.tight_layout()
plt.show()