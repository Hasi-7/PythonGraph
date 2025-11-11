import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from matplotlib.lines import Line2D

# ----------------------------
# Models
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
fname = 'QFactorGraph5(67cm).txt'   # <- change if needed
raw = np.loadtxt(fname, skiprows=1)

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
time_sigma  = 1.0 / 30.0   # one frame at 30 fps
angle_sigma = 0.005        # rad

# ============================================================
#              ROBUST PEAK DETECTION PIPELINE
# ============================================================

# 1) Estimate and remove a slow baseline (midline).
#    Use Savitzky–Golay with a window much larger than the period.
#    We'll set this window AFTER we estimate the period.
dt = float(np.median(np.diff(t)))
if not np.isfinite(dt) or dt <= 0:
    raise SystemExit("Non-positive or NaN sampling interval.")

# Rough period estimate from zero-crossings of the raw signal
def estimate_period_from_zeros(t, y):
    y_nz = np.where(y == 0.0, np.finfo(float).eps, y)
    sgn  = np.sign(y_nz)
    zc   = np.where(sgn[:-1] * sgn[1:] < 0)[0]  # sign changes
    if zc.size < 3:
        return None
    # Every two zero-crossings ~ one period; take median
    t_z = t[zc]
    # For more robustness, use spacing between alternate crossings
    if t_z.size >= 3:
        alt = t_z[2:] - t_z[:-2]
        alt = alt[alt > 0]
        T_est = float(np.median(alt)) if alt.size else None
    else:
        T_est = None
    return T_est

T_est = estimate_period_from_zeros(t, y)
if T_est is None or not np.isfinite(T_est):
    # fallback: FFT-based guess
    y_demean = y - np.median(y)
    n = len(y_demean)
    if n < 8:
        raise SystemExit("Too few samples to estimate period.")
    f = np.fft.rfftfreq(n, d=dt)
    Y = np.abs(np.fft.rfft(y_demean))
    # ignore DC
    if f.size > 1:
        k = 1 + np.argmax(Y[1:])
        if f[k] > 0:
            T_est = 1.0 / f[k]
        else:
            T_est = 1.5
    else:
        T_est = 1.5

T_est = float(T_est)
print(f"Estimated period T_est ≈ {T_est:.4f} s")

# SavGol window: at least ~3 periods, odd, and not longer than the record
win_pts = int(max(9, np.floor(3.0 * T_est / dt) // 2 * 2 + 1))  # make odd
win_pts = min(win_pts, len(y) - (1 - len(y) % 2))  # must be <= len(y) and odd
if win_pts < 9:  # safeguard
    win_pts = 9
poly = 2 if win_pts >= 11 else 1
baseline = savgol_filter(y, window_length=win_pts, polyorder=poly, mode='interp')

# Baseline-relative signal (this is the true oscillatory part)
y_rel = y - baseline

# 2) Find ALL LOCAL MAXIMA throughout the entire dataset
# Minimum distance between adjacent peaks in samples
min_dist = int(max(1, np.floor(0.8 * T_est / dt)))   # 80% of period in samples

# Noise estimate from the last chunk (where amplitude is smallest)
tail = y_rel[int(0.8 * len(y_rel)):]
mad  = np.median(np.abs(tail - np.median(tail))) if tail.size else np.median(np.abs(y_rel))
sigma_noise = 1.4826 * mad if mad > 0 else np.std(tail) if tail.size else np.std(y_rel)

# Use a VERY low prominence to catch all maxima, even small ones near the end
prominence = max(1.0 * sigma_noise, 0.5 * angle_sigma)  # very low threshold to catch all peaks

# Find ONLY POSITIVE peaks (local maxima) - this is what we want!
peak_idx, props = find_peaks(y_rel, distance=min_dist, prominence=prominence)

print(f"Initial peaks found: {peak_idx.size}")

# If we didn't find enough peaks, try with even lower settings
if peak_idx.size < 10:
    print("Not enough peaks found, trying with lower prominence...")
    prominence_low = 0.3 * angle_sigma
    min_dist_low = int(max(1, np.floor(0.7 * T_est / dt)))
    peak_idx, props = find_peaks(y_rel, distance=min_dist_low, prominence=prominence_low)
    print(f"Second attempt found: {peak_idx.size} peaks")

peak_idx = np.sort(peak_idx)
have_peaks = peak_idx.size >= 3
if not have_peaks:
    print("Warning: <3 maxima found — results may be limited.")
else:
    print(f"Found {peak_idx.size} local maxima in the data")

# Keep arrays of peak times and amplitudes
t_peaks = t[peak_idx] if have_peaks else np.array([])
# For maxima, take the actual values (they should already be positive for maxima)
A_peaks = y_rel[peak_idx] if have_peaks else np.array([])

# Simple sanity check: remove any peaks that are actually negative (shouldn't happen with proper maxima)
if have_peaks and len(A_peaks) > 0:
    positive_mask = A_peaks > 0
    if not np.all(positive_mask):
        print(f"Removing {np.sum(~positive_mask)} negative 'maxima'")
        peak_idx = peak_idx[positive_mask]
        t_peaks = t_peaks[positive_mask]
        A_peaks = A_peaks[positive_mask]
        have_peaks = len(peak_idx) >= 3

# ============================================================
#     FITS (damped cosine; envelope on baseline-relative A)
# ============================================================

# Initial guesses for the damped cosine on the baseline-relative series
def safe_initial_guess(t_full, y_full, t_pk):
    A_guess   = max(np.max(np.abs(y_full[:min(200, len(y_full))])) if y_full.size else 1.0, 1e-3)
    if t_pk.size >= 3:
        dtp = np.diff(t_pk)
        dtp = dtp[dtp > 0]
        T_guess = float(np.median(dtp)) if dtp.size else T_est
    else:
        T_guess = T_est
    T_guess   = max(T_guess, 1e-3)
    tau_guess = max((t_full[-1] - t_full[0]) * 3.0, 1.0)
    # phase guess from first sample sign & slope
    c = np.clip(y_full[0] / A_guess, -1.0, 1.0)
    phi_guess = float(np.arccos(c))
    dy0 = np.gradient(y_full, t_full, edge_order=2)[0]
    if dy0 > 0:  # choose descending from a maximum
        phi_guess = -phi_guess
    return [A_guess, tau_guess, T_guess, phi_guess]

theta0_g, tau_g, T_g, phi_g = safe_initial_guess(t, y_rel, t_peaks)
initial_guess = [theta0_g, tau_g, T_g, phi_g]
lower = [0.0,            0.1 * tau_g,  0.5 * T_g, -2*np.pi]
upper = [2.0 * abs(theta0_g) + 1e-6, 10.0 * tau_g, 1.5 * T_g,  2*np.pi]

try:
    params, pcov = curve_fit(
        theta_func, t, y_rel,
        p0=initial_guess, bounds=(lower, upper), maxfev=200000
    )
    fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
    perr = np.sqrt(np.diag(pcov))
except Exception as e:
    print("Damped cosine fit did not fully converge:", e)
    params = np.array(initial_guess, float)
    pcov = np.full((4, 4), np.nan)
    fitted_theta_0, fitted_tau, fitted_T, fitted_phi_0 = params
    perr = np.array([np.nan, np.nan, np.nan, np.nan])

print("\nDamped-cosine parameters on baseline-relative data (±1σ):")
print(f"  theta0 = {fitted_theta_0:.6g} ± {perr[0] if np.isfinite(perr[0]) else np.nan:.2g} rad")
print(f"  tau    = {fitted_tau:.6g} ± {perr[1] if np.isfinite(perr[1]) else np.nan:.2g} s")
print(f"  T      = {fitted_T:.6g} ± {perr[2] if np.isfinite(perr[2]) else np.nan:.2g} s")
print(f"  phi0   = {fitted_phi_0:.6g} ± {perr[3] if np.isfinite(perr[3]) else np.nan:.2g} rad")

# Envelope fit on baseline-relative maxima (A_peaks)
# Using the formula: θ(t) = θ₀ * exp(-t/τ)
if have_peaks and len(A_peaks) >= 3:
    print(f"\nFitting exponential envelope to {len(A_peaks)} maxima...")
    
    # Use ALL detected maxima for the fit
    t_fit = t_peaks
    A_fit_data = A_peaks
    
    # Get initial guess from linear fit in log space: ln(A) = ln(θ₀) - t/τ
    with np.errstate(divide='ignore', invalid='ignore'):
        log_A = np.log(np.clip(A_fit_data, 1e-12, None))
    
    # Remove any inf or nan values
    valid_log = np.isfinite(log_A)
    if np.sum(valid_log) >= 3:
        # Linear fit: log(A) = b - m*t, where m = 1/tau and b = log(theta_0)
        m_lin, b_lin = np.polyfit(t_fit[valid_log], log_A[valid_log], 1)
        
        # Extract initial parameters
        if m_lin < 0:  # Should be negative for decay
            tau_seed = -1.0 / m_lin  # tau = -1/slope
        else:
            tau_seed = (t_fit[-1] - t_fit[0]) / 2.0  # fallback: half the time range
        
        theta0_seed = np.exp(b_lin)  # θ₀ = exp(intercept)
        
        # Sanity checks
        theta0_seed = max(theta0_seed, A_fit_data[0] * 0.5)
        theta0_seed = min(theta0_seed, A_fit_data[0] * 2.0)
        tau_seed = max(tau_seed, 1.0)
        tau_seed = min(tau_seed, (t_fit[-1] - t_fit[0]) * 5.0)
    else:
        theta0_seed = A_fit_data[0]
        tau_seed = (t_fit[-1] - t_fit[0]) / 2.0
    
    exp_guess = [theta0_seed, tau_seed]
    print(f"Initial guess: θ₀={theta0_seed:.6f}, τ={tau_seed:.6f}")
    
    # Set reasonable bounds for the fit
    exp_lower = [0.1 * theta0_seed, 0.1 * tau_seed]
    exp_upper = [5.0 * theta0_seed, 20.0 * tau_seed]
    
    try:
        # Fit the exponential decay: A(t) = θ₀ * exp(-t/τ)
        exp_params, exp_pcov = curve_fit(
            exp_decay, t_fit, A_fit_data,
            p0=exp_guess,
            bounds=(exp_lower, exp_upper),
            maxfev=100000
        )
        A_fit, tau_fit = exp_params
        exp_perr = np.sqrt(np.diag(exp_pcov))
        sigma_tau = exp_perr[1]
        print(f"Exponential fit SUCCESS: θ₀={A_fit:.6f}±{exp_perr[0]:.6f}, τ={tau_fit:.6f}±{sigma_tau:.6f} s")
        
        # Calculate R² to check fit quality
        A_predicted = exp_decay(t_fit, A_fit, tau_fit)
        ss_res = np.sum((A_fit_data - A_predicted)**2)
        ss_tot = np.sum((A_fit_data - np.mean(A_fit_data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R² = {r_squared:.6f}")
        
    except Exception as e:
        print("Envelope fit did not converge:", e)
        # Use the linear fit results as fallback
        A_fit, tau_fit = exp_guess
        exp_perr = np.array([np.nan, np.nan])
        sigma_tau = np.nan
else:
    print("Skipping envelope fit: insufficient peaks.")
    A_fit, tau_fit, exp_perr, sigma_tau = np.nan, np.nan, np.array([np.nan, np.nan]), np.nan

if have_peaks:
    print("\nExponential envelope on baseline-relative peaks (±1σ):")
    print(f"  A   = {A_fit:.6g} ± {exp_perr[0] if np.isfinite(exp_perr[0]) else np.nan:.2g} rad")
    print(f"  tau = {tau_fit:.6g} ± {sigma_tau if np.isfinite(sigma_tau) else np.nan:.2g} s")

# ============================================================
#                 4% TARGET COUNT (with σ)
#     using baseline-relative amplitudes at peaks
# ============================================================
if have_peaks:
    A0 = A_peaks[0]
    target = 0.04 * A0

    sigma_A = angle_sigma * np.ones_like(A_peaks)
    A_upper = A_peaks + sigma_A
    A_lower = A_peaks - sigma_A

    # bracket where the 4% target is reached
    idx_lo = np.where(A_upper <= target)[0]  # definitely below
    idx_hi = np.where(A_lower <= target)[0]  # possibly below

    N_lo = idx_lo[0] + 1 if idx_lo.size else None
    N_hi = idx_hi[0] + 1 if idx_hi.size else None

    band_lo = 0.99 * target
    band_hi = 1.01 * target
    in_band = (A_peaks >= band_lo) & (A_peaks <= band_hi)
    overlaps = (A_lower <= band_hi) & (A_upper >= band_lo)
    highlight_mask = in_band | overlaps
    t_4 = t_peaks[highlight_mask]
    A_4 = A_peaks[highlight_mask]

    if (N_lo is not None) and (N_hi is not None):
        N_est  = 0.5 * (N_lo + N_hi)
        sigma_N = 0.5 * abs(N_hi - N_lo)
        Q_count = N_est          # NOTE: at 4%, Q_count = N (no ×2)
        sigma_Q_count = sigma_N
        print(f"\n4% target amplitude = {target:.6f} rad (relative to baseline)")
        print(f"First definite crossing (A_upper <= target): N_lo = {N_lo}")
        print(f"First possible crossing (A_lower <= target): N_hi = {N_hi}")
        print(f"N_count (to 4%) = {N_est:.2f} ± {sigma_N:.2f}")
        print(f"Q_count (oscillations to 4%) = {Q_count:.2f} ± {sigma_Q_count:.2f}")
    else:
        if A_peaks.size:
            k = int(np.argmin(np.abs(A_peaks - target)))
            N_est = k + 1
            sigma_N = 1.0
            Q_count = N_est
            sigma_Q_count = sigma_N
            t_4 = t_peaks[[k]]
            A_4 = A_peaks[[k]]
            print("\nCould not bracket 4% with error bars; using nearest peak.")
            print(f"N_count (to 4%) ≈ {N_est} ± {sigma_N}")
            print(f"Q_count (oscillations to 4%) ≈ {Q_count:.2f} ± {sigma_Q_count:.2f}")
else:
    t_4 = A_4 = np.array([])
    target = np.nan
    Q_count = sigma_Q_count = np.nan

# ============================================================
#     Mean period from FIRST 10 oscillations (± uncertainty)
# ============================================================
if have_peaks and t_peaks.size >= 2:
    N_needed = min(11, t_peaks.size)     # 10 periods need 11 maxima
    periods  = np.diff(t_peaks[:N_needed])
    N_used   = periods.size
    if N_used >= 1:
        T_bar = float(np.mean(periods))
        sigma_time_mean   = (np.sqrt(2.0) * time_sigma) / np.sqrt(N_used)
        sigma_sample_mean = np.std(periods, ddof=1) / np.sqrt(N_used) if N_used >= 2 else 0.0
        sigma_T_bar = float(np.sqrt(sigma_time_mean**2 + sigma_sample_mean**2))
        t_start = t_peaks[0]
        t_end   = t_peaks[N_used]
        print(f"\nMean period (first {N_used} oscillations): T̄ = {T_bar:.6g} ± {sigma_T_bar:.2g} s")
        print(f"Used maxima window: start t = {t_start:.3f} s, end t = {t_end:.3f} s")
    else:
        T_bar = sigma_T_bar = np.nan
        print("\nNot enough maxima to form periods from the start window.")
else:
    T_bar = sigma_T_bar = np.nan
    print("\nNot enough maxima to compute the first-10 average.")

# ============================================================
#                         Q2
# ============================================================
if np.isfinite(tau_fit) and np.isfinite(T_bar):
    Q2 = np.pi * tau_fit / T_bar
    dQ_dtau = np.pi / T_bar
    dQ_dT   = -np.pi * tau_fit / (T_bar**2)
    sigma_Q2 = np.sqrt((dQ_dtau * sigma_tau)**2 + (dQ_dT * sigma_T_bar)**2)
    print(f"Q2 = {Q2:.6g} ± {sigma_Q2:.2g}")
else:
    Q2 = sigma_Q2 = np.nan

# ============================================================
#                         PLOTS
# ============================================================
plt.figure(figsize=(12, 6))

# raw data with error bars
plt.errorbar(
    t, y,
    xerr=time_sigma, yerr=angle_sigma,
    fmt='o', markersize=2.0, color='tab:red', alpha=0.5,
    ecolor='lightgray', elinewidth=0.6, capsize=0,
    label='Measured data (±σ)', zorder=1
)

# overlay damped cosine on baseline-relative (shift back to baseline for display)
t_plot = np.linspace(t.min(), t.max(), 3000)
y_fit_rel = theta_func(t_plot, *params)
y_fit = y_fit_rel + np.interp(t_plot, t, baseline)
plt.plot(t_plot, y_fit, color='tab:blue', linewidth=2.0, label='Damped-cosine fit', zorder=4)

# exponential envelope - plot as smooth curve using fitted exponential only
# The envelope represents the maximum amplitude, so we calculate where the baseline is at peak times
# and plot a smooth exponential curve through those peak positions
if have_peaks and np.isfinite(A_fit):
    # Calculate the average baseline value at peak locations for a smooth reference
    baseline_at_peaks = baseline[peak_idx]
    baseline_avg = np.mean(baseline_at_peaks)
    
    # Create smooth exponential envelope curve
    env_rel = exp_decay(t_plot, A_fit, tau_fit)
    env_abs = env_rel + baseline_avg  # Use constant baseline for smooth curve
    plt.plot(t_plot, env_abs, linestyle='--', linewidth=2.5, color='magenta',
             label='Exponential fit (relative)', zorder=3)

# show detected peaks at their absolute positions (baseline + amplitude with sign)
if have_peaks:
    # reconstruct the actual peak values for plotting
    sign_at_peaks = np.sign(y_rel[peak_idx])
    y_peaks_abs = baseline[peak_idx] + sign_at_peaks * A_peaks
    plt.scatter(t_peaks, y_peaks_abs, s=25, color='tab:green', label='Pendulum peaks', zorder=5)

plt.xlabel('Time (s)', fontsize=24)
plt.ylabel(r'Angle $\theta(t)$ [rad]', fontsize=24)
plt.title('Amplitude Vs. Time - Cosine and Exponential Fits', fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.grid(True, alpha=0.25)
plt.legend(loc='upper right', frameon=True, fontsize=16)
plt.tight_layout()
plt.show()

# Maxima-only view + residuals of envelope (baseline-relative)
if have_peaks:
    A_fit_at_peaks = exp_decay(t_peaks, A_fit, tau_fit) if np.isfinite(A_fit) else np.zeros_like(t_peaks)
    env_residuals  = A_peaks - A_fit_at_peaks

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 2]})

    ax1.errorbar(
        t_peaks, A_peaks,
        xerr=time_sigma, yerr=angle_sigma,
        fmt='o', ms=4, color='tab:blue', ecolor='lightgray',
        alpha=0.85, label='Maxima (±σ)', zorder=2
    )
    if np.isfinite(A_fit):
        ax1.plot(t_peaks, A_fit_at_peaks, 'g--', lw=2.0, label='Exponential fit', zorder=3)

    if t_4.size > 0:
        ax1.scatter(t_4, A_4, s=90, facecolor='orange', edgecolor='black', linewidth=1.0,
                    label='4% amplitude peaks (±1% band + σ overlap)', zorder=5)

    ax1.set_ylabel('Peak amplitude above baseline [rad]', fontsize=20)
    ax1.set_title('Exponential Fit on Baseline-Relative Maxima', fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc='upper right', frameon=True, fontsize=14)

    ax2.axhline(0, color='k', lw=1)
    ax2.errorbar(
        t_peaks, env_residuals,
        xerr=time_sigma, yerr=angle_sigma,
        fmt='o', ms=3, ecolor='lightgray', elinewidth=1,
        capsize=0, color='black', label='Residuals (data - fit)'
    )
    ax2.set_xlabel('Time (s)', fontsize=20)
    ax2.set_ylabel('Residual [rad]', fontsize=20)
    ax2.set_title('Residuals of Exponential Fit (relative)', fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(True, axis='y', linestyle=':')
    ax2.legend(loc='upper right', frameon=True, fontsize=14)

    fig.tight_layout()
    plt.show()