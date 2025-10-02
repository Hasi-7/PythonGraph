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
# Load data
# ----------------------------
data = np.loadtxt('QFactorGraph2.txt', skiprows=1)
theta_data = data[:, 0]  # angle in radians
t_data = data[:, 1]      # time in seconds

print("First 5 theta_data:", theta_data[:5])
print("First 5 t_data:", t_data[:5])

# ----------------------------
# Measurement uncertainties
# ----------------------------
# Time uncertainty = 1 frame at 30 fps
time_sigma = 1/30
# Angle uncertainty estimate (based on 720p video, 2.5 ft camera, pendulum length ~0.536 m)
angle_sigma = 0.002  # ~0.1 deg ≈ 0.002 rad, rough estimate

# ----------------------------
# Initial guess + fit damped cosine
# ----------------------------
initial_guess = [theta_data[0], 200, 1.5, 0]  # amplitude, tau, T, phi0
fit_kwargs = dict(p0=initial_guess, maxfev=20000)

params, pcov = curve_fit(theta_func, t_data, theta_data, **fit_kwargs)
fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
perr = np.sqrt(np.diag(pcov))
print("Best fit damped cosine (±1σ):")
print(f"  theta0 = {fitted_theta_0:.4f} ± {perr[0]:.2g}")
print(f"  tau    = {fitted_tau:.4f} ± {perr[1]:.2g} s")
print(f"  T      = {fitted_T:.4f} ± {perr[2]:.2g} s")
print(f"  phi0   = {fitted_phi_0:.4f} ± {perr[3]:.2g} rad")

# ----------------------------
# Find peaks for envelope
# ----------------------------
peaks, _ = find_peaks(theta_data, prominence=0.001)
have_peaks = peaks.size > 0

if have_peaks:
    t_peaks = t_data[peaks]
    theta_peaks = theta_data[peaks]
    A_peaks = np.abs(theta_peaks)

    # Envelope fit
    exp_guess = [np.max(A_peaks), 200]
    exp_params, exp_pcov = curve_fit(exp_decay, t_peaks, A_peaks,
                                     p0=exp_guess, maxfev=20000)
    A_fit, tau_fit = exp_params
    exp_perr = np.sqrt(np.diag(exp_pcov))
    print("Exponential envelope fit (±1σ):")
    print(f"  A   = {A_fit:.4f} ± {exp_perr[0]:.2g} rad")
    print(f"  tau = {tau_fit:.4f} ± {exp_perr[1]:.2g} s")
else:
    print("No peaks found — skipping envelope fit.")
    t_peaks = theta_peaks = A_peaks = []
    tau_fit = A_fit = exp_perr = None

# ----------------------------
# Define 20% amplitude threshold
# ----------------------------
if have_peaks:
    A0 = A_peaks[0]
    thresh = 0.20 * A0

    # Find window where amplitude crosses 20%
    crossing = np.where(A_peaks <= thresh)[0]
    if crossing.size > 0:
        k_cross = crossing[0]
        t_left = t_peaks[k_cross-1] if k_cross > 0 else t_peaks[k_cross]
        t_right = t_peaks[k_cross]
    else:
        t_left, t_right = None, None
        print("Warning: amplitude never fell below 20%.")

    # Count peaks inside window
    if t_left and t_right:
        in_window = (t_peaks >= t_left) & (t_peaks <= t_right)
        n_peaks_in_window = int(np.sum(in_window))
        print(f"Peaks inside 20% window [{t_left:.2f}, {t_right:.2f}] s: {n_peaks_in_window}")

# ----------------------------
# Plot 1: Main fit with data
# ----------------------------
plt.figure(figsize=(12, 6))
plt.errorbar(t_data, theta_data, xerr=time_sigma, yerr=angle_sigma,
             fmt='o', markersize=2.2, color='tab:red', alpha=0.55,
             ecolor='lightgray', elinewidth=0.8, capsize=0,
             label='Measured data (±σ)', zorder=1)

t_plot = np.linspace(t_data.min(), t_data.max(), 2000)

if have_peaks:
    plt.plot(t_plot, exp_decay(t_plot, A_fit, tau_fit),
             linestyle='--', linewidth=2.5, color='magenta',
             label='Exponential envelope fit', zorder=3)
    plt.scatter(t_peaks, theta_peaks, s=18, color='tab:green',
                label='Envelope peaks', zorder=2)

plt.plot(t_plot, theta_func(t_plot, *params),
         color='tab:blue', linewidth=2.2, label='Damped-cosine fit', zorder=4)

plt.xlabel('Time (s)')
plt.ylabel(r'Angle $\theta(t)$ [rad]')
plt.title('Damped Pendulum: Data, Cosine Fit, and Envelope')
plt.grid(True, alpha=0.25)
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 2: 20% threshold window
# ----------------------------
if have_peaks and t_left and t_right:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(t_peaks, A_peaks, s=22, color='tab:blue', alpha=0.85, label='|Peak amplitude|')

    ax.hlines(thresh, t_peaks.min(), t_peaks.max(),
              colors='magenta', linestyles=':', linewidth=2, label='20% of initial peak')

    ax.scatter(t_peaks[in_window], A_peaks[in_window], s=30, color='orange',
               edgecolor='k', linewidth=0.6, label='Peaks in 20% window')

    ax.axvspan(t_left, t_right, color='orange', alpha=0.22, label='Crossing window')

    ax.plot(t_peaks, exp_decay(t_peaks, A_fit, tau_fit), 'g--', alpha=0.8, label='Envelope at peaks')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Peak amplitude $|A|$ [rad]')
    ax.set_title('20% Amplitude Crossing Window (peaks counted in orange)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    plt.show()

    # ----------------------------
    # Plot 3: Envelope fit on peaks + residuals
    # ----------------------------
    A_fit_at_peaks = exp_decay(t_peaks, A_fit, tau_fit)
    env_residuals = A_peaks - A_fit_at_peaks

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 2]})

    ax1.scatter(t_peaks, A_peaks, s=22, color='tab:blue', alpha=0.85, label='|Peak amplitude|')
    ax1.plot(t_peaks, A_fit_at_peaks, 'g--', lw=2.2, label='Exponential fit (peaks)')
    ax1.hlines(thresh, t_peaks.min(), t_peaks.max(), colors='magenta', linestyles=':', lw=2,
               label='20% of initial peak')
    ax1.set_ylabel(r'Peak amplitude $|A|$ [rad]')
    ax1.set_title('Exponential Envelope Fit on Peaks')
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper right', frameon=True)

    ax2.axhline(0, color='k', lw=1)
    ax2.errorbar(t_peaks, env_residuals, yerr=angle_sigma,
                 fmt='o', ms=3, ecolor='lightgray', elinewidth=1,
                 capsize=0, color='black', label='Residuals')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residual (rad)')
    ax2.set_title('Residuals: peaks − exponential fit')
    ax2.grid(True, axis='y', linestyle=':')
    ax2.legend(loc='upper right', frameon=True)

    fig.tight_layout()
    plt.show()
