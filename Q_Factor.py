import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.lines import Line2D

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
# Load data (auto-detect which col is time)
# ----------------------------
raw = np.loadtxt('QFactorGraph4(New Pendulum).txt', skiprows=1)

col0_inc = np.all(np.diff(raw[:, 0]) >= -1e-12)
col1_inc = np.all(np.diff(raw[:, 1]) >= -1e-12)

if col0_inc and not col1_inc:
    t_data     = raw[:, 0].astype(float)
    theta_data = raw[:, 1].astype(float)
elif col1_inc and not col0_inc:
    t_data     = raw[:, 1].astype(float)
    theta_data = raw[:, 0].astype(float)
else:
    # fallback (assume first col is time)
    t_data     = raw[:, 0].astype(float)
    theta_data = raw[:, 1].astype(float)

# Remove any duplicate/NaN rows and sort by time
m = np.isfinite(theta_data) & np.isfinite(t_data)
t_data, theta_data = t_data[m], theta_data[m]
mask_unique = np.r_[True, np.diff(t_data) > 0]
t_data, theta_data = t_data[mask_unique], theta_data[mask_unique]
order = np.argsort(t_data)
t_data, theta_data = t_data[order], theta_data[order]

print("First 5 theta_data:", theta_data[:5])
print("First 5 t_data:", t_data[:5])

# Shift time to start at zero (helps optimizer with phase)
t0 = float(t_data[0])
t = t_data - t0
y = theta_data

# ----------------------------
# Measurement uncertainties
# ----------------------------
time_sigma  = 1.0 / 30.0   # one timestamp uncertainty: 1 frame at 30 fps
angle_sigma = 0.005        # rad

# ----------------------------
# Robust peak detection (pos → neg → abs)
# ----------------------------
pk_pos, _ = find_peaks(y, prominence=1e-3)
if pk_pos.size >= 3:
    peaks = pk_pos
else:
    pk_min, _ = find_peaks(-y, prominence=1e-3)
    if pk_min.size >= 3:
        peaks = pk_min
    else:
        pk_abs, _ = find_peaks(np.abs(y), prominence=5e-4)
        peaks = pk_abs

have_peaks = peaks.size >= 3
if not have_peaks:
    print("Warning: still found <3 peaks; results may be limited.")

t_peaks     = t[peaks] if have_peaks else np.array([])
theta_peaks = y[peaks] if have_peaks else np.array([])
A_peaks     = np.abs(theta_peaks) if have_peaks else np.array([])

# ----------------------------
# Safe initial guesses for the damped cosine
# ----------------------------
def safe_initial_guess(t_full, y_full, t_pk, y_pk):
    # defaults
    A_guess   = max(np.max(np.abs(y_full[:200])) if y_full.size else 1.0, 1e-3)
    T_guess   = 1.5
    tau_guess = max((t_full[-1] - t_full[0]) * 3.0, 1.0)

    # Try period from peak spacing
    if t_pk.size >= 3:
        dtp = np.diff(t_pk)
        dtp = dtp[dtp > 0]
        if dtp.size:
            T_guess = float(np.median(dtp))

    # Final clamp so it's never zero/negative
    T_guess = max(T_guess, 1e-3)

    # Phase guess: use first sample sign & slope
    c = np.clip(y_full[0] / A_guess, -1.0, 1.0)
    phi_guess = float(np.arccos(c))
    dy0 = np.gradient(y_full, t_full, edge_order=2)[0]
    if dy0 > 0:
        phi_guess = -phi_guess

    return [A_guess, tau_guess, T_guess, phi_guess]

theta0_g, tau_g, T_g, phi_g = safe_initial_guess(t, y, t_peaks, theta_peaks)
initial_guess = [theta0_g, tau_g, T_g, phi_g]
print("Initial guesses:", initial_guess)

# Bounds built from clamped guesses
lower = [0.0,            0.1 * tau_g,  0.5 * T_g, -2*np.pi]
upper = [2.0 * abs(theta0_g) + 1e-6, 10.0 * tau_g, 1.5 * T_g,  2*np.pi]

# ----------------------------
# Damped cosine fit (with bounds & fallback)
# ----------------------------
try:
    params, pcov = curve_fit(
        theta_func, t, y,
        p0=initial_guess, bounds=(lower, upper),
        maxfev=200000
    )
    fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
    perr = np.sqrt(np.diag(pcov))
except Exception as e:
    print("Damped cosine fit did not fully converge:", e)
    params = np.array(initial_guess, float)
    pcov = np.full((4, 4), np.nan)
    fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
    perr = np.array([np.nan, np.nan, np.nan, np.nan])

print("\nDamped-cosine parameters (±1σ):")
print(f"  theta0 = {fitted_theta_0:.6g} ± {perr[0] if np.isfinite(perr[0]) else np.nan:.2g} rad")
print(f"  tau    = {fitted_tau:.6g} ± {perr[1] if np.isfinite(perr[1]) else np.nan:.2g} s")
print(f"  T      = {fitted_T:.6g} ± {perr[2] if np.isfinite(perr[2]) else np.nan:.2g} s")
print(f"  phi0   = {fitted_phi_0:.6g} ± {perr[3] if np.isfinite(perr[3]) else np.nan:.2g} rad")

# ----------------------------
# Envelope fit on maxima (|theta| at same-kind extrema)
# ----------------------------
if have_peaks:
    exp_guess  = [np.max(A_peaks), max(1.0, tau_g)]
    try:
        exp_params, exp_pcov = curve_fit(
            exp_decay, t_peaks, A_peaks,
            p0=exp_guess,
            bounds=([0.5*exp_guess[0], 0.1*exp_guess[1]],
                    [2.0*exp_guess[0], 10.0*exp_guess[1]]),
            maxfev=100000
        )
        A_fit, tau_fit = exp_params
        exp_perr = np.sqrt(np.diag(exp_pcov))
        sigma_tau = exp_perr[1]
    except Exception as e:
        print("Envelope fit did not fully converge:", e)
        A_fit, tau_fit = exp_guess
        exp_perr = np.array([np.nan, np.nan])
        sigma_tau = np.nan
else:
    print("Skipping envelope fit: insufficient peaks.")
    A_fit, tau_fit, exp_perr, sigma_tau = np.nan, np.nan, np.array([np.nan, np.nan]), np.nan

if have_peaks:
    print("\nExponential envelope (±1σ):")
    print(f"  A   = {A_fit:.6g} ± {exp_perr[0] if np.isfinite(exp_perr[0]) else np.nan:.2g} rad")
    print(f"  tau = {tau_fit:.6g} ± {sigma_tau if np.isfinite(sigma_tau) else np.nan:.2g} s")

# ----------------------------
# 20% points (STRICT ±1%), maxima only
# ----------------------------
if have_peaks:
    A0     = A_peaks[0]
    target = 0.20 * A0
    tol    = 0.01 * target
    near20_mask = (A_peaks >= target - tol) & (A_peaks <= target + tol)

    idx20 = np.where(near20_mask)[0]
    if idx20.size > 0:
        first_20_idx   = int(idx20[0])
        N_count        = first_20_idx + 1
        sigma_N_count  = 1
        Q_count        = 2 * N_count
        sigma_Q_count  = 2
    else:
        first_20_idx = None
        N_count = sigma_N_count = Q_count = sigma_Q_count = None

    t_20 = t_peaks[near20_mask]
    A_20 = A_peaks[near20_mask]

    print(f"\n20% target amplitude = {target:.6f} rad, band = [{target - tol:.6f}, {target + tol:.6f}] rad")
    print(f"Number of maxima within ±1% of target: {t_20.size}")
    if first_20_idx is not None:
        print(f"N_count (to first 20%) = {N_count} ± {sigma_N_count}")
        print(f"Q_count = 2·N_count = {Q_count} ± {sigma_Q_count}")
    else:
        print("No maxima within ±1% of the 20% target were found.")
else:
    t_20 = A_20 = np.array([])
    target = tol = np.nan

# ----------------------------
# Mean period from total elapsed time and its uncertainty
# ----------------------------
if have_peaks and t_peaks.size >= 2:
    N_periods = t_peaks.size - 1
    t_start   = t_peaks[0]
    t_end     = t_peaks[-1]
    t_total   = t_end - t_start
    T_bar     = t_total / N_periods
    sigma_T_bar = (np.sqrt(2.0) * time_sigma) / N_periods

    print(f"\nPendulum run start time = {t_start:.3f} s")
    print(f"Pendulum run end time   = {t_end:.3f} s (last full oscillation peak)")
    print(f"Total elapsed time      = {t_total:.3f} s")
    print(f"\nMean period T̄ = {T_bar:.6g} ± {sigma_T_bar:.2g} s  (from N={N_periods} periods)")
else:
    T_bar = sigma_T_bar = np.nan

# ----------------------------
# Q2 and its propagated uncertainty
# ----------------------------
if np.isfinite(tau_fit) and np.isfinite(T_bar):
    Q2 = np.pi * tau_fit / T_bar
    dQ_dtau = np.pi / T_bar
    dQ_dT   = -np.pi * tau_fit / (T_bar**2)
    sigma_Q2 = np.sqrt((dQ_dtau * sigma_tau)**2 + (dQ_dT * sigma_T_bar)**2)
    print(f"Q2 = {Q2:.6g} ± {sigma_Q2:.2g}")
else:
    Q2 = sigma_Q2 = np.nan

# ----------------------------
# Plot 1: Main time series + cosine & envelope fits
# ----------------------------
plt.figure(figsize=(12, 6))
plt.errorbar(
    t, y,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', markersize=2.2, color='tab:red', alpha=0.55,
    ecolor='lightgray', elinewidth=0.8, capsize=0,
    label='Measured data (±σ)', zorder=1
)

t_plot = np.linspace(t.min(), t.max(), 2000)

if np.isfinite(A_fit) and np.isfinite(tau_fit):
    plt.plot(
        t_plot, exp_decay(t_plot, A_fit, tau_fit),
        linestyle='--', linewidth=2.5, color='magenta',
        label='Exponential fit', zorder=3
    )

if have_peaks:
    plt.errorbar(
        t_peaks, theta_peaks,
        xerr=time_sigma, yerr=angle_sigma,
        fmt='o', markersize=4, color='tab:green',
        ecolor='lightgray', capsize=0,
        label='Pendulum peaks', zorder=2
    )

theta_fit = theta_func(t_plot, *params)
plt.plot(t_plot, theta_fit, color='tab:blue', linewidth=2.2, label='Damped-cosine fit', zorder=4)

plt.xlabel('Time (s)', fontsize=36)
plt.ylabel(r'Angle $\theta(t)$ [rad]', fontsize=36)
plt.title('Amplitude Vs. Time - Cosine and Exponential Fits', fontsize=38)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.grid(True, alpha=0.25)

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
# Plot 2: Maxima-only — envelope fit + residuals (+ orange markers)
# ----------------------------
if have_peaks:
    A_fit_at_peaks = exp_decay(t_peaks, A_fit, tau_fit) if np.isfinite(A_fit) else np.zeros_like(t_peaks)
    env_residuals  = A_peaks - A_fit_at_peaks

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 2]})

    ax1.errorbar(t_peaks, A_peaks, xerr=time_sigma, yerr=angle_sigma,
                 fmt='o', ms=4, color='tab:blue', ecolor='lightgray',
                 alpha=0.85, label='Maxima (±σ)', zorder=2)
    if np.isfinite(A_fit):
        ax1.plot(t_peaks, A_fit_at_peaks, 'g--', lw=2.2, label='Exponential fit', zorder=3)

    if t_20.size > 0:
        ax1.scatter(t_20, A_20, s=90, facecolor='orange', edgecolor='black', linewidth=1.1,
                    label='20% amplitude peaks (±1%)', zorder=5)

    ax1.set_ylabel('Peak amplitude [rad]', fontsize=36)
    ax1.set_title('Exponential Fit on Maxima (20% amplitude peaks highlighted)', fontsize=38)
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.grid(True, alpha=0.25)

    legend_handles2 = [
        Line2D([0], [0], linestyle='--', color='green', lw=2.8, label='Exponential fit'),
        Line2D([0], [0], marker='o', color='orange', markeredgecolor='black', linestyle='None', markersize=12, label='20% amplitude peaks (±1%)'),
        Line2D([0], [0], marker='o', color='tab:blue', linestyle='None', markersize=11, label='Maxima (±σ)'),
    ]
    ax1.legend(handles=legend_handles2, loc='upper right', frameon=True, fontsize=24)

    ax2.axhline(0, color='k', lw=1)
    ax2.errorbar(t_peaks, env_residuals, xerr=time_sigma, yerr=angle_sigma,
                 fmt='o', ms=3, ecolor='lightgray', elinewidth=1,
                 capsize=0, color='black', label='Residuals (data - fit)')
    ax2.set_xlabel('Time (s)', fontsize=36)
    ax2.set_ylabel('Residual [rad]', fontsize=36)
    ax2.set_title('Residuals of Exponential Fit', fontsize=38)
    ax2.tick_params(axis='both', which='major', labelsize=28)
    ax2.grid(True, axis='y', linestyle=':')
    ax2.legend(loc='upper right', frameon=True, fontsize=24)

    fig.tight_layout()
    plt.show()
else:
    print("Not enough peaks for the maxima/residuals plot; only time series shown.")