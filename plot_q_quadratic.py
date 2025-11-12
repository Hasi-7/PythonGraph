# Generate Q-factor vs Length graph with quadratic fit only

import fit_black_box as bb
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Load Q-factor data
filename_q = "length_Q_consolidated.txt"
length_q, length_q_err, q_factor, q_err = bb.load_data(filename_q)

# Define quadratic model
def quadratic(x, a, b, c):
    return a + b * x + c * x**2

# Fit quadratic model
init_guess = (60, 0, 0)
popt, pcov = optimize.curve_fit(quadratic, length_q, q_factor, sigma=q_err, 
                                p0=init_guess, absolute_sigma=True)
puncert = np.sqrt(np.diagonal(pcov))

# Calculate statistics
residuals = q_factor - quadratic(length_q, *popt)
chi_squared = np.sum((residuals / q_err)**2)
dof = len(q_factor) - len(popt)
reduced_chi_squared = chi_squared / dof

ss_res = np.sum(residuals**2)
ss_tot = np.sum((q_factor - np.mean(q_factor))**2)
r_squared = 1 - (ss_res / ss_tot)

print("="*70)
print("QUADRATIC FIT FOR Q-FACTOR VS LENGTH")
print("="*70)
print(f"\nModel: Q = a + b*L + c*L²")
print(f"\nBest fit parameters:")
print(f"  a = {popt[0]:.2f} ± {puncert[0]:.2f}")
print(f"  b = {popt[1]:.2f} ± {puncert[1]:.2f}")
print(f"  c = {popt[2]:.2f} ± {puncert[2]:.2f}")
print(f"\nStatistics:")
print(f"  χ² = {chi_squared:.2f}")
print(f"  Reduced χ² = {reduced_chi_squared:.2f} (dof={dof})")
print(f"  R² = {r_squared:.4f}")
print()

# Create the plot
plt.rcParams.update({'font.size': 32})
plt.rcParams['figure.figsize'] = 14, 12

start_q = min(length_q)
stop_q = max(length_q)
xs_q = np.linspace(start_q, stop_q, 1000)
curve_q = quadratic(xs_q, *popt)

fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

# Main plot
ax1.errorbar(length_q, q_factor, yerr=q_err, xerr=length_q_err, 
             fmt="o", label="Measured data", color="green", markersize=10)
ax1.plot(xs_q, curve_q, 
         label="Quadratic fit", 
         color="red", linewidth=3)
ax1.legend(loc='best', fontsize=28)
ax1.set_xlabel("Length (m)", fontsize=36)
ax1.set_ylabel("Q-factor", fontsize=36)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=30)

# Residuals plot
ax2.errorbar(length_q, residuals, yerr=q_err, xerr=length_q_err, 
             fmt="o", color="green", markersize=10)
ax2.axhline(y=0, color="black", linestyle='--', linewidth=1)
ax2.set_xlabel("Length (m)", fontsize=36)
ax2.set_ylabel("Residuals", fontsize=36)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=30)

fig.tight_layout()
plt.savefig("q_factor_quadratic_fit.png", dpi=300)
print("Saved: q_factor_quadratic_fit.png")
plt.show()
