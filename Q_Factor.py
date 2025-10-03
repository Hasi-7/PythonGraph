import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.lines import Line2D  # for custom, larger legend markers

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
time_sigma  = 1.0 / 30.0   # one timestamp uncertainty: 1 frame at 30 fps
angle_sigma = 0.005        # 0.005 rad

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

print("\nExponential fit (±1σ):")
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
    first_20_idx   = int(idx20[0])
    N_count        = first_20_idx + 1      # oscillations to FIRST ~20% peak
    sigma_N_count  = 1                     # ±1 oscillation ambiguity
    Q_count        = 2 * N_count
    sigma_Q_count  = 2                     # ±2 from ±1 oscillation
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
N_periods = t_peaks.size - 1
t_start   = t_peaks[0]
t_end     = t_peaks[-1]        # last peak = end of last full oscillation
t_total   = t_end - t_start
T_bar     = t_total / N_periods

# σ(T̄) from two timestamps: σ_t_total = sqrt(2) σ_t ; divide by N_periods
sigma_T_bar = (np.sqrt(2.0) * time_sigma) / N_periods

print(f"\nPendulum run start time = {t_start:.3f} s")
print(f"Pendulum run end time   = {t_end:.3f} s (last full oscillation peak)")
print(f"Total elapsed time      = {t_total:.3f} s")
print(f"\nMean period T̄ = {T_bar:.6g} ± {sigma_T_bar:.2g} s  (from N={N_periods} periods)")

# ----------------------------
# Q2 and its propagated uncertainty
# ----------------------------
Q2 = np.pi * tau_fit / T_bar
dQ_dtau = np.pi / T_bar
dQ_dT   = -np.pi * tau_fit / (T_bar**2)
sigma_Q2 = np.sqrt((dQ_dtau * sigma_tau)**2 + (dQ_dT * sigma_T_bar)**2)

print(f"Q2 = {Q2:.6g} ± {sigma_Q2:.2g}")

# ----------------------------
# Plot 1: Main time series + cosine fit (NO orange markers here)
# ----------------------------
plt.figure(figsize=(12, 6))
# Keep graph markers the same as before
plt.errorbar(
    t_data, theta_data,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', markersize=2.2, color='tab:red', alpha=0.55,
    ecolor='lightgray', elinewidth=0.8, capsize=0,
    label='Measured data (±σ)', zorder=1
)

t_plot = np.linspace(t_data.min(), t_data.max(), 2000)

plt.plot(
    t_plot, exp_decay(t_plot, A_fit, tau_fit),
    linestyle='--', linewidth=2.5, color='magenta',
    label='Exponential fit', zorder=3
)

plt.errorbar(
    t_peaks, theta_peaks,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', markersize=4, color='tab:green',
    ecolor='lightgray', capsize=0,
    label='Pendulum peaks', zorder=2
)

plt.plot(
    t_plot, theta_func(t_plot, *params),
    color='tab:blue', linewidth=2.2, label='Damped-cosine fit', zorder=4
)

plt.xlabel('Time (s)', fontsize=36)
plt.ylabel(r'Angle $\theta(t)$ [rad]', fontsize=36)
plt.title('Amplitude Vs. Time - Cosine and Exponential Fits', fontsize=38)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.grid(True, alpha=0.25)

# ---- Legend with BIGGER markers ONLY (plot markers unchanged)
legend_handles = [
    Line2D([0], [0], linestyle='--', color='magenta', lw=2.8, label='Exponential fit'),
    Line2D([0], [0], linestyle='-',  color='tab:blue', lw=2.8, label='Damped-cosine fit'),
    Line2D([0], [0], marker='o', color='tab:red',   linestyle='None', markersize=11, label='Measured data (±σ)'),
    Line2D([0], [0], marker='o', color='tab:green', linestyle='None', markersize=11, label='Pendulum peaks'),
]
plt.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=24)
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

# Keep plot marker sizes as before
ax1.errorbar(
    t_peaks, A_peaks,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', ms=4, color='tab:blue', ecolor='lightgray',
    alpha=0.85, label='Maxima (±σ)', zorder=2
)
ax1.plot(t_peaks, A_fit_at_peaks, 'g--', lw=2.2, label='Exponential fit', zorder=3)

# ORANGE points plotted LAST with high zorder to ensure they sit on top
if t_20.size > 0:
    ax1.scatter(
        t_20, A_20,
        s=90, facecolor='orange', edgecolor='black', linewidth=1.1,
        label='20% amplitude peaks (±1%)', zorder=5
    )

ax1.set_ylabel('Peak amplitude [rad]', fontsize=36)
ax1.set_title('Exponential Fit on Maxima (20% amplitude peaks highlighted)', fontsize=38)
ax1.tick_params(axis='both', which='major', labelsize=28)
ax1.grid(True, alpha=0.25)

# Legend with bigger markers ONLY
legend_handles2 = [
    Line2D([0], [0], linestyle='--', color='green', lw=2.8, label='Exponential fit'),
    Line2D([0], [0], marker='o', color='orange', markeredgecolor='black', linestyle='None', markersize=12, label='20% amplitude peaks (±1%)'),
    Line2D([0], [0], marker='o', color='tab:blue', linestyle='None', markersize=11, label='Maxima (±σ)'),
]
ax1.legend(handles=legend_handles2, loc='upper right', frameon=True, fontsize=24)

ax2.axhline(0, color='k', lw=1)
ax2.errorbar(
    t_peaks, env_residuals,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', ms=3, ecolor='lightgray', elinewidth=1,
    capsize=0, color='black', label='Residuals (data - fit)'
)
ax2.set_xlabel('Time (s)', fontsize=36)
ax2.set_ylabel('Residual [rad]', fontsize=36)
ax2.set_title('Residuals of Exponential Fit', fontsize=38)
ax2.tick_params(axis='both', which='major', labelsize=28)
ax2.grid(True, axis='y', linestyle=':')
ax2.legend(loc='upper right', frameon=True, fontsize=24)

fig.tight_layout()
plt.show()