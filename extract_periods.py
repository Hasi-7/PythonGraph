import numpy as np
from scipy.signal import find_peaks, savgol_filter

# File mapping: filename -> length (in cm from filename)
files_to_process = {
    'QFactorGraph3(71cm).txt': 71,
    'QFactorGraph5(67cm).txt': 67,
    'QFactorGraph6(64cm).txt': 64,
    'QFactorGraph7(59cm).txt': 59,
    'QFactorGraph8(55cm).txt': 55,
    'QFactorGraph9(48cm).txt': 48,
    'QFactorGraph10(43cm).txt': 43,
}

def calculate_period_from_file(fname):
    """Calculate average period from first 10 oscillations"""
    # Load data
    raw = np.loadtxt(fname, skiprows=1)
    
    # Auto-detect which column is time
    col0_inc = np.all(np.diff(raw[:, 0]) >= -1e-12)
    col1_inc = np.all(np.diff(raw[:, 1]) >= -1e-12)
    
    if col0_inc and not col1_inc:
        t_data = raw[:, 0].astype(float)
        theta_data = raw[:, 1].astype(float)
    elif col1_inc and not col0_inc:
        t_data = raw[:, 1].astype(float)
        theta_data = raw[:, 0].astype(float)
    else:
        t_data = raw[:, 0].astype(float)
        theta_data = raw[:, 1].astype(float)
    
    # Clean / sort
    m = np.isfinite(theta_data) & np.isfinite(t_data)
    t_data, theta_data = t_data[m], theta_data[m]
    mask_unique = np.r_[True, np.diff(t_data) > 0]
    t_data, theta_data = t_data[mask_unique], theta_data[mask_unique]
    order = np.argsort(t_data)
    t_data, theta_data = t_data[order], theta_data[order]
    
    # Shift time to start at zero
    t0 = float(t_data[0])
    t = t_data - t0
    y = theta_data
    
    # Estimate period
    dt = float(np.median(np.diff(t)))
    
    # Rough period estimate from zero-crossings
    y_nz = np.where(y == 0.0, np.finfo(float).eps, y)
    sgn = np.sign(y_nz)
    zc = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    
    if zc.size >= 3:
        t_z = t[zc]
        if t_z.size >= 3:
            alt = t_z[2:] - t_z[:-2]
            alt = alt[alt > 0]
            T_est = float(np.median(alt)) if alt.size else 1.5
        else:
            T_est = 1.5
    else:
        # FFT fallback
        y_demean = y - np.median(y)
        n = len(y_demean)
        f = np.fft.rfftfreq(n, d=dt)
        Y = np.abs(np.fft.rfft(y_demean))
        if f.size > 1:
            k = 1 + np.argmax(Y[1:])
            if f[k] > 0:
                T_est = 1.0 / f[k]
            else:
                T_est = 1.5
        else:
            T_est = 1.5
    
    # Remove baseline
    win_pts = int(max(9, np.floor(3.0 * T_est / dt) // 2 * 2 + 1))
    win_pts = min(win_pts, len(y) - (1 - len(y) % 2))
    if win_pts < 9:
        win_pts = 9
    poly = 2 if win_pts >= 11 else 1
    baseline = savgol_filter(y, window_length=win_pts, polyorder=poly, mode='interp')
    y_rel = y - baseline
    
    # Find peaks
    min_dist = int(max(1, np.floor(0.8 * T_est / dt)))
    tail = y_rel[int(0.8 * len(y_rel)):]
    mad = np.median(np.abs(tail - np.median(tail))) if tail.size else np.median(np.abs(y_rel))
    sigma_noise = 1.4826 * mad if mad > 0 else np.std(tail) if tail.size else np.std(y_rel)
    prominence = max(1.0 * sigma_noise, 0.005 * 0.5)
    
    peak_idx, _ = find_peaks(y_rel, distance=min_dist, prominence=prominence)
    
    if peak_idx.size < 10:
        # Try with lower settings
        prominence_low = 0.3 * 0.005
        min_dist_low = int(max(1, np.floor(0.7 * T_est / dt)))
        peak_idx, _ = find_peaks(y_rel, distance=min_dist_low, prominence=prominence_low)
    
    t_peaks = t[peak_idx]
    A_peaks = y_rel[peak_idx]
    
    # Remove negative peaks
    positive_mask = A_peaks > 0
    t_peaks = t_peaks[positive_mask]
    
    # Calculate average period from first 10 oscillations
    if t_peaks.size >= 2:
        N_needed = min(11, t_peaks.size)
        periods = np.diff(t_peaks[:N_needed])
        N_used = periods.size
        
        if N_used >= 1:
            T_bar = float(np.mean(periods))
            sigma_sample = np.std(periods, ddof=1) / np.sqrt(N_used) if N_used >= 2 else 0.0
            time_sigma = 1.0 / 30.0  # one frame at 30 fps
            sigma_time_mean = (np.sqrt(2.0) * time_sigma) / np.sqrt(N_used)
            sigma_T_bar = float(np.sqrt(sigma_time_mean**2 + sigma_sample**2))
            
            return T_bar, sigma_T_bar, N_used
    
    return None, None, 0

# Process all files
results = []
print("Processing files...")
print("="*60)

for fname, length_cm in sorted(files_to_process.items(), key=lambda x: x[1], reverse=True):
    print(f"\nProcessing: {fname}")
    T_bar, sigma_T, n_periods = calculate_period_from_file(fname)
    
    if T_bar is not None:
        length_m = length_cm / 100.0  # Convert cm to meters
        results.append((length_m, T_bar, sigma_T))
        print(f"  Length: {length_cm} cm = {length_m:.3f} m")
        print(f"  Periods used: {n_periods}")
        print(f"  Average Period: {T_bar:.6f} ± {sigma_T:.5f} s")
    else:
        print(f"  ERROR: Could not calculate period")

# Sort by length (ascending)
results.sort(key=lambda x: x[0])

# Save to file
output_file = 'length_period_consolidated.txt'
with open(output_file, 'w') as f:
    f.write("# Length (m) vs Average Period (s) from first 10 oscillations\n")
    f.write("# Length(m)  Period(s)  Uncertainty(s)\n")
    for length, period, sigma in results:
        f.write(f"{length:.4f}  {period:.6f}  {sigma:.6f}\n")

print("\n" + "="*60)
print(f"\nResults saved to: {output_file}")
print("\nFinal data:")
print("Length (m)  Period (s)  Uncertainty (s)")
print("-" * 45)
for length, period, sigma in results:
    print(f"{length:.4f}      {period:.6f}     ±{sigma:.6f}")
