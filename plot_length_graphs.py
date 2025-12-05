# Script to create graphs for Period vs Length and Q-factor vs Length

import fit_black_box as bb
import numpy as np

# Define fitting functions
def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * x**b

def linear(x, m, b):
    """Linear: y = m*x + b"""
    return m * x + b

def quadratic(x, a, b, c):
    """Quadratic: y = a + b*x + c*x^2"""
    return a + b * x + c * x**2

# ===================================================================
# GRAPH 1: Period vs Length (log-log scale)
# ===================================================================
print("="*70)
print("GENERATING PERIOD VS LENGTH GRAPH (LOG-LOG SCALE)")
print("="*70)

# Load period data (4 columns: length, length_err, period, period_err)
filename_period = "length_period_consolidated.txt"
length, length_err, period, period_err = bb.load_data(filename_period)

# For period vs length, we expect T ∝ √L, so T = a * L^0.5
# Initial guess: a ≈ 2 (since T ≈ 2π√(L/g) and √(1/g) ≈ 0.32)
init_guess_period = (2.0, 0.5)

# Plot with log-log scale
print("\nFitting power law: T = a * L^b")
print("Expected: b ≈ 0.5 (from simple pendulum formula T = 2π√(L/g))")
print()

# Modify the plot_fit function call to enable log scale
import matplotlib.pyplot as plt
import scipy.optimize as optimize

plt.rcParams.update({'font.size': 32})
plt.rcParams['figure.figsize'] = 14, 12

popt, pcov = optimize.curve_fit(power_law, length, period, sigma=period_err, 
                                 p0=init_guess_period, absolute_sigma=True)
puncert = np.sqrt(np.diagonal(pcov))

print("Best fit parameters:")
print(f"  a = {popt[0]:.6f} ± {puncert[0]:.6f}")
print(f"  b = {popt[1]:.6f} ± {puncert[1]:.6f}")
print()

# Create the plot
start = min(length)
stop = max(length)
xs = np.logspace(np.log10(start), np.log10(stop), 1000)
curve = power_law(xs, *popt)

fig1, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

ax1.errorbar(length, period, yerr=period_err, xerr=length_err, 
             fmt="o", label="Measured data", color="blue", markersize=10)
ax1.plot(xs, curve, label=f"Best fit: T = {popt[0]:.2f}L$^{{{popt[1]:.2f}}}$", 
         color="red", linewidth=3)
ax1.legend(loc='upper left', fontsize=28)
ax1.set_xlabel("Length (m)", fontsize=36)
ax1.set_ylabel("Period (s)", fontsize=36)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both')
ax1.tick_params(labelsize=30)

# Residuals
residual = period - power_law(length, *popt)
ax2.errorbar(length, residual, yerr=period_err, xerr=length_err, 
             fmt="o", color="blue", markersize=10)
ax2.axhline(y=0, color="black", linestyle='--', linewidth=1)
ax2.set_xlabel("Length (m)", fontsize=36)
ax2.set_ylabel("Residuals (s)", fontsize=36)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=30)

fig1.tight_layout()
plt.savefig("period_vs_length_loglog.png", dpi=300)
print("Saved: period_vs_length_loglog.png")
plt.show()

# ===================================================================
# GRAPH 2: Q-factor vs Length (linear scale)
# ===================================================================
print("\n" + "="*70)
print("GENERATING Q-FACTOR VS LENGTH GRAPH")
print("="*70)

# Load Q-factor data (4 columns: length, length_err, q_factor, q_err)
filename_q = "length_Q_consolidated.txt"
length_q, length_q_err, q_factor, q_err = bb.load_data(filename_q)

# Define additional model functions
def constant(x, c):
    """Constant: y = c"""
    return c * np.ones_like(x)

def exponential(x, a, b):
    """Exponential: y = a * exp(b*x)"""
    return a * np.exp(b * x)

# Try different models and compare them
models = {
    'Constant': (constant, (70,), lambda p: f"Q = {p[0]:.2f}"),
    'Linear': (linear, (0, 70), lambda p: f"Q = {p[0]:.1f}L + {p[1]:.1f}"),
    'Quadratic': (quadratic, (60, 0, 0), lambda p: f"Q = {p[0]:.1f} + {p[1]:.1f}L + {p[2]:.1f}L²"),
    'Exponential': (exponential, (60, 0.5), lambda p: f"Q = {p[0]:.1f}exp({p[1]:.3f}L)")
}

print("\nTrying different models for Q-factor vs Length:")
print("="*70)

results = {}
for model_name, (func, init_guess, label_func) in models.items():
    try:
        popt_model, pcov_model = optimize.curve_fit(func, length_q, q_factor, sigma=q_err, 
                                                     p0=init_guess, absolute_sigma=True)
        puncert_model = np.sqrt(np.diagonal(pcov_model))
        
        # Calculate chi-squared and reduced chi-squared
        residuals = q_factor - func(length_q, *popt_model)
        chi_squared = np.sum((residuals / q_err)**2)
        dof = len(q_factor) - len(popt_model)  # degrees of freedom
        reduced_chi_squared = chi_squared / dof
        
        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((q_factor - np.mean(q_factor))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results[model_name] = {
            'func': func,
            'popt': popt_model,
            'puncert': puncert_model,
            'label_func': label_func,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'r_squared': r_squared,
            'dof': dof
        }
        
        print(f"\n{model_name} Model:")
        print(f"  Parameters: {popt_model}")
        print(f"  Uncertainties: {puncert_model}")
        print(f"  χ² = {chi_squared:.4f}")
        print(f"  Reduced χ² = {reduced_chi_squared:.4f} (dof={dof})")
        print(f"  R² = {r_squared:.6f}")
    except Exception as e:
        print(f"\n{model_name} Model: FAILED - {str(e)}")

# Find the best model based on reduced chi-squared (closest to 1 is ideal)
print("\n" + "="*70)
print("MODEL COMPARISON:")
print("="*70)
for model_name, result in results.items():
    print(f"{model_name:12s}: Reduced χ² = {result['reduced_chi_squared']:.4f}, R² = {result['r_squared']:.6f}")

best_model = min(results.items(), key=lambda x: abs(x[1]['reduced_chi_squared'] - 1.0))
best_name = best_model[0]
best_result = best_model[1]

print(f"\nBest model (by reduced χ²): {best_name}")
print(f"  Reduced χ² = {best_result['reduced_chi_squared']:.4f}")
print(f"  R² = {best_result['r_squared']:.6f}")

# Create the plot with the best model
print("\n" + "="*70)
print(f"Plotting with {best_name} model")
print("="*70)

start_q = min(length_q)
stop_q = max(length_q)
xs_q = np.linspace(start_q, stop_q, 1000)
curve_q = best_result['func'](xs_q, *best_result['popt'])

fig2, (ax1_q, ax2_q) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

ax1_q.errorbar(length_q, q_factor, yerr=q_err, xerr=length_q_err, 
               fmt="o", label="Measured data", color="green", markersize=10)
ax1_q.plot(xs_q, curve_q, 
           label=f"Best fit ({best_name}): {best_result['label_func'](best_result['popt'])}", 
           color="red", linewidth=3)
ax1_q.legend(loc='best', fontsize=28)
ax1_q.set_xlabel("Length (m)", fontsize=36)
ax1_q.set_ylabel("Q-factor", fontsize=36)
ax1_q.grid(True, alpha=0.3)
ax1_q.tick_params(labelsize=30)
ax1_q.set_title(f"Reduced χ² = {best_result['reduced_chi_squared']:.3f}, R² = {best_result['r_squared']:.4f}", 
                fontsize=28)

# Residuals
residual_q = q_factor - best_result['func'](length_q, *best_result['popt'])
ax2_q.errorbar(length_q, residual_q, yerr=q_err, xerr=length_q_err, 
               fmt="o", color="green", markersize=10)
ax2_q.axhline(y=0, color="black", linestyle='--', linewidth=1)
ax2_q.set_xlabel("Length (m)", fontsize=36)
ax2_q.set_ylabel("Residuals", fontsize=36)
ax2_q.grid(True, alpha=0.3)
ax2_q.tick_params(labelsize=30)

fig2.tight_layout()
plt.savefig("q_factor_vs_length.png", dpi=300)
print("Saved: q_factor_vs_length.png")
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nPeriod vs Length:")
print(f"  Fitted exponent b = {popt[1]:.4f} ± {puncert[1]:.4f}")
print(f"  Theoretical value = 0.5 (for simple pendulum)")
print(f"  Difference: {abs(popt[1] - 0.5):.4f} ({abs(popt[1] - 0.5)/puncert[1]:.2f}σ)")
print("\nQ-factor vs Length:")
print(f"  Q-factor ranges from {min(q_factor):.1f} to {max(q_factor):.1f}")
print(f"  Mean Q-factor: {np.mean(q_factor):.1f} ± {np.std(q_factor):.1f}")
