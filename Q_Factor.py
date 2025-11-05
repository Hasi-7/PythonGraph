import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
# Load data (auto-detect which column is time)
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
    t_data     = raw[:, 0].astype(float)
    theta_data = raw[:, 1].astype(float)

# Clean / sort
m = np.isfinite(theta_data) & np.isfinite(t_data)
t_data, theta_data = t_data[m], theta_data[m]
mask_unique = np.r_[True, np.diff(t_data) > 0]
t_data, theta_data = t_data[mask_unique], theta_data[mask_unique]
order = np.argsort(t_data)
t_data, theta_data = t_data[order], theta_data[order]

print("First 5 theta_data:", theta_data[:5])
print("First 5 t_data:", t_data[:5])

# Shift time to start at zero
t0 = float(t_data[0])
t = t_data - t0
y = theta_data

# ----------------------------
# Measurement uncertainties
# ----------------------------
time_sigma  = 1.0/30.0   # timestamp (one frame at 30 fps)
angle_sigma = 0.005      # ~0.29°

# ----------------------------
# Robust maxima via zero-crossing gating
#   - find indices where θ changes sign
#   - in each half-cycle, take the single largest |θ|
#   - keep only positive (else negative) extrema so they're same-kind
# ----------------------------
def halfcycle_extrema_indices(y):
    # sign changes (ignore exact zeros by nudging)
    y_nz = np.where(y == 0.0, np.finfo(float).eps, y)
    sign = np.sign(y_nz)
    zc = np.where(sign[:-1] * sign[1:] < 0)[0]  # indices where sign flips
    # segment bounds (half-cycles)
    starts = np.r_[0, zc + 1]
    ends   = np.r_[zc, len(y) - 1]
    ext_idx = []
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        seg = np.abs(y[s:e+1])
        k = np.argmax(seg)
        ext_idx.append(s + k)
    return np.array(sorted(set(ext_idx)))

ext_all = halfcycle_extrema_indices(y)
pos_ext = ext_all[y[ext_all] > 0]
neg_ext = ext_all[y[ext_all] < 0]
# choose the richer set so they're same-kind maxima
if pos_ext.size >= 3:
    peak_idx = pos_ext
elif neg_ext.size >= 3:
    peak_idx = neg_ext
else:
    peak_idx = np.array([], dtype=int)

have_peaks = peak_idx.size >= 3
if not have_peaks:
    print("Warning: <3 same-sign maxima found — results may be limited.")

t_peaks     = t[peak_idx] if have_peaks else np.array([])
theta_peaks = y[peak_idx] if have_peaks else np.array([])
A_peaks     = np.abs(theta_peaks) if have_peaks else np.array([])

# ----------------------------
# Safe initial guesses for the damped cosine
# ----------------------------
def safe_initial_guess(t_full, y_full, t_pk):
    A_guess   = max(np.max(np.abs(y_full[:200])) if y_full.size else 1.0, 1e-3)
    # period guess: spacing between same-kind maxima
    if t_pk.size >= 3:
        dtp = np.diff(t_pk)
        dtp = dtp[dtp > 0]
        T_guess = float(np.median(dtp)) if dtp.size else 1.5
    else:
        T_guess = 1.5
    T_guess   = max(T_guess, 1e-3)
    tau_guess = max((t_full[-1] - t_full[0]) * 3.0, 1.0)
    # phase guess from first sample sign & slope
    c = np.clip(y_full[0] / A_guess, -1.0, 1.0)
    phi_guess = float(np.arccos(c))
    dy0 = np.gradient(y_full, t_full, edge_order=2)[0]
    if dy0 > 0:
        phi_guess = -phi_guess
    return [A_guess, tau_guess, T_guess, phi_guess]

theta0_g, tau_g, T_g, phi_g = safe_initial_guess(t, y, t_peaks)
initial_guess = [theta0_g, tau_g, T_g, phi_g]
print("Initial guesses:", initial_guess)

# Bounds from guesses
lower = [0.0,            0.1 * tau_g,  0.5 * T_g, -2*np.pi]
upper = [2.0 * abs(theta0_g) + 1e-6, 10.0 * tau_g, 1.5 * T_g,  2*np.pi]

# ----------------------------
# Damped cosine fit (bounded; robust fallback)
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
# Envelope fit (on same-kind maxima)
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
        exp_perr  = np.sqrt(np.diag(exp_pcov))
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
# 4% points using error bars (bracket N and compute σN) + ±1% band around 4%
# ----------------------------
if have_peaks:
    A0     = A_peaks[0]
    target = 0.04 * A0                         # 4% of initial amplitude

    # Peak-wise y-uncertainty (same as plotted)
    sigma_A = angle_sigma * np.ones_like(A_peaks)

    # Error-bar envelopes
    A_upper = A_peaks + sigma_A
    A_lower = A_peaks - sigma_A

    # ---- Bracket N using the target value (for Q_count) ----
    # First DEFINITE crossing: even the upper bar is below target
    idx_lo = np.where(A_upper <= target)[0]
    N_lo = idx_lo[0] + 1 if idx_lo.size else None

    # First POSSIBLE crossing: lower bar touches/drops below target
    idx_hi = np.where(A_lower <= target)[0]
    N_hi = idx_hi[0] + 1 if idx_hi.size else None

    # ---- Mark peaks "within ±1% of the 4% target" OR whose error bars overlap that band ----
    band_lo = 0.99 * target
    band_hi = 1.01 * target
    in_band_nominal = (A_peaks >= band_lo) & (A_peaks <= band_hi)
    overlaps_band   = (A_lower <= band_hi) & (A_upper >= band_lo)
    highlight_mask  = in_band_nominal | overlaps_band

    t_4 = t_peaks[highlight_mask]
    A_4 = A_peaks[highlight_mask]

    if (N_lo is not None) and (N_hi is not None):
        # Best estimate and uncertainty from bracket
        N_est  = 0.5 * (N_lo + N_hi)
        sigma_N = 0.5 * abs(N_hi - N_lo)

        # Q_count is the oscillation count to reach 4% (NO x2 factor)
        Q_count       = N_est
        sigma_Q_count = sigma_N

        print(f"\n4% target amplitude = {target:.6f} rad")
        print(f"First definite crossing (A_upper <= target): N_lo = {N_lo}")
        print(f"First possible crossing (A_lower <= target): N_hi = {N_hi}")
        print(f"N_count (to 4%) = {N_est:.2f} ± {sigma_N:.2f}")
        print(f"Q_count (oscillations to 4%) = {Q_count:.2f} ± {sigma_Q_count:.2f}")
        print(f"Peaks within ±1% band OR overlapping it via error bars: {t_4.size}")
    else:
        # Fallback: nearest-peak estimate if we can’t bracket with error bars
        if A_peaks.size:
            k = int(np.argmin(np.abs(A_peaks - target)))
            N_est = k + 1
            sigma_N = 1.0
            Q_count       = N_est
            sigma_Q_count = sigma_N
            t_4 = t_peaks[[k]]
            A_4 = A_peaks[[k]]
            print("\nCould not bracket 4% with error bars; using nearest peak.")
            print(f"N_count (to 4%) ≈ {N_est} ± {sigma_N}")
            print(f"Q_count (oscillations to 4%) ≈ {Q_count:.2f} ± {sigma_Q_count:.2f}")
        else:
            t_4 = A_4 = np.array([])
            Q_count = sigma_Q_count = np.nan
            print("\nNo peaks → cannot compute 4% crossing and Q_count.")
else:
    t_4 = A_4 = np.array([])
    target = np.nan
    Q_count = sigma_Q_count = np.nan

# ----------------------------
# Mean period from FIRST 10 OSCILLATIONS (+ uncertainty)
# ----------------------------
if have_peaks and t_peaks.size >= 2:
    # Need 11 maxima for 10 periods; use what's available
    N_needed  = min(11, t_peaks.size)
    periods   = np.diff(t_peaks[:N_needed])
    N_used    = periods.size

    if N_used >= 1:
        T_bar = float(np.mean(periods))
        # timing-limited SE on the mean (two timestamps per period)
        sigma_time_mean = (np.sqrt(2.0) * time_sigma) / np.sqrt(N_used)
        # sample scatter SE
        if N_used >= 2:
            sigma_sample_mean = np.std(periods, ddof=1) / np.sqrt(N_used)
        else:
            sigma_sample_mean = 0.0
        sigma_T_bar = float(np.sqrt(sigma_time_mean**2 + sigma_sample_mean**2))

        t_start = t_peaks[0]
        t_end   = t_peaks[N_used]  # last max used for the first-10 average
        print(f"\nMean period (first {N_used} oscillations): T̄ = {T_bar:.6g} ± {sigma_T_bar:.2g} s")
        print(f"Used maxima window: start t = {t_start:.3f} s, end t = {t_end:.3f} s")
    else:
        T_bar = sigma_T_bar = np.nan
        print("\nNot enough maxima to form periods from the start window.")
else:
    T_bar = sigma_T_bar = np.nan
    print("\nNot enough maxima to compute the first-10 average.")

# ----------------------------
# Q2 (uses tau from envelope and T̄ from first-10 periods)
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
    plt.plot(t_plot, exp_decay(t_plot, A_fit, tau_fit),
             linestyle='--', linewidth=2.5, color='magenta',
             label='Exponential fit', zorder=3)

if have_peaks:
    plt.errorbar(t_peaks, theta_peaks,
                 xerr=time_sigma, yerr=angle_sigma,
                 fmt='o', markersize=4, color='tab:green',
                 ecolor='lightgray', capsize=0,
                 label='Pendulum peaks', zorder=2)

theta_fit = theta_func(t_plot, *params)
plt.plot(t_plot, theta_fit, color='tab:blue', linewidth=2.2, label='Damped-cosine fit', zorder=4)

plt.xlabel('Time (s)', fontsize=36)
plt.ylabel(r'Angle $\theta(t)$ [rad]', fontsize=36)
plt.title('Amplitude Vs. Time - Cosine and Exponential Fits', fontsize=38)
plt.tick_params(axis='both', which='major', labelsize=28)
plt.grid(True, alpha=0.25)
plt.legend(handles=[
    Line2D([0], [0], linestyle='--', color='magenta', lw=2.8, label='Exponential fit'),
    Line2D([0], [0], linestyle='-',  color='tab:blue', lw=2.8, label='Damped-cosine fit'),
    Line2D([0], [0], marker='o', color='tab:red',   linestyle='None', markersize=11, label='Measured data (±σ)'),
    Line2D([0], [0], marker='o', color='tab:green', linestyle='None', markersize=11, label='Pendulum peaks'),
], loc='upper right', frameon=True, fontsize=24)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot 2: Maxima-only — envelope fit + residuals (+ 4% overlap markers)
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

    # Highlight peaks whose error bars overlap the ±1% band around the 4% target
    if t_4.size > 0:
        ax1.scatter(t_4, A_4, s=90, facecolor='orange', edgecolor='black', linewidth=1.1,
                    label='4% amplitude peaks (±1% band + σ overlap)', zorder=5)

    ax1.set_ylabel('Peak amplitude [rad]', fontsize=36)
    ax1.set_title('Exponential Fit on Maxima (4% amplitude peaks highlighted)', fontsize=38)
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper right', frameon=True, fontsize=24)

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