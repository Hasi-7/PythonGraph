# Compare all Q-factor models side by side

import fit_black_box as bb
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# Load Q-factor data
filename_q = "length_Q_consolidated.txt"
length_q, length_q_err, q_factor, q_err = bb.load_data(filename_q)

# Define model functions
def constant(x, c):
    return c * np.ones_like(x)

def linear(x, m, b):
    return m * x + b

def quadratic(x, a, b, c):
    return a + b * x + c * x**2

def cubic(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def exponential(x, a, b):
    return a * np.exp(b * x)

def power_law(x, a, b):
    return a * x**b

def logarithmic(x, a, b):
    return a * np.log(x) + b

def inverse(x, a, b):
    return a / x + b

def sqrt_model(x, a, b):
    return a * np.sqrt(x) + b

def gaussian(x, A, mu, sigma, offset):
    """Gaussian peak/valley: A * exp(-(x-mu)^2/(2*sigma^2)) + offset"""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def lorentzian(x, A, x0, gamma, offset):
    """Lorentzian peak/valley: A * gamma^2 / ((x-x0)^2 + gamma^2) + offset"""
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + offset

def sinusoidal(x, A, omega, phi, offset):
    """Sinusoidal: A * sin(omega*x + phi) + offset"""
    return A * np.sin(omega * x + phi) + offset

def double_exponential(x, a1, b1, a2, b2):
    """Double exponential: a1*exp(b1*x) + a2*exp(b2*x)"""
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x)

def rational(x, a, b, c, d):
    """Rational function: (a + b*x) / (c + d*x)"""
    return (a + b * x) / (c + d * x)

# Fit all models
models = {
    'Constant': (constant, (70,), "Q = {:.2f}"),
    'Linear': (linear, (0, 70), "Q = {:.1f}L + {:.1f}"),
    'Quadratic': (quadratic, (60, 0, 0), "Q = {:.1f} + {:.1f}L + {:.1f}L²"),
    'Cubic': (cubic, (60, 0, 0, 0), "Q = {:.1f} + {:.1f}L + {:.1f}L² + {:.1f}L³"),
    'Exponential': (exponential, (60, 0.5), "Q = {:.1f}exp({:.3f}L)"),
    'Power Law': (power_law, (70, 0.5), "Q = {:.1f}L^{:.3f}"),
    'Logarithmic': (logarithmic, (10, 60), "Q = {:.1f}ln(L) + {:.1f}"),
    'Inverse': (inverse, (0.1, 70), "Q = {:.3f}/L + {:.1f}"),
    'Square Root': (sqrt_model, (50, 30), "Q = {:.1f}√L + {:.1f}"),
    'Gaussian': (gaussian, (-20, 0.6, 0.1, 80), "Q = {:.1f}exp(-(L-{:.3f})²/2σ²) + {:.1f}"),
    'Lorentzian': (lorentzian, (-20, 0.6, 0.1, 80), "Q = {:.1f}γ²/((L-{:.3f})²+γ²) + {:.1f}"),
    'Sinusoidal': (sinusoidal, (15, 10, 0, 70), "Q = {:.1f}sin({:.2f}L + {:.2f}) + {:.1f}"),
    'Dbl Exponential': (double_exponential, (30, 1, 40, -0.5), "Q = {:.1f}e^({:.2f}L) + {:.1f}e^({:.2f}L)"),
    'Rational': (rational, (40, 20, 0.5, 0.01), "Q = ({:.1f}+{:.1f}L)/({:.2f}+{:.3f}L)")
}

results = {}
for model_name, (func, init_guess, label_template) in models.items():
    try:
        popt, pcov = optimize.curve_fit(func, length_q, q_factor, sigma=q_err, 
                                        p0=init_guess, absolute_sigma=True, maxfev=10000)
        puncert = np.sqrt(np.diagonal(pcov))
        
        residuals = q_factor - func(length_q, *popt)
        chi_squared = np.sum((residuals / q_err)**2)
        dof = len(q_factor) - len(popt)
        reduced_chi_squared = chi_squared / dof
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((q_factor - np.mean(q_factor))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results[model_name] = {
            'func': func,
            'popt': popt,
            'puncert': puncert,
            'label_template': label_template,
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'r_squared': r_squared,
            'residuals': residuals
        }
    except Exception as e:
        print(f"Warning: {model_name} model failed to converge: {str(e)}")

# Create comparison plot
fig, axes = plt.subplots(4, 4, figsize=(20, 18))
axes = axes.flatten()

colors = ['red', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'olive', 
          'magenta', 'lime', 'navy', 'teal', 'maroon']
xs_q = np.linspace(min(length_q), max(length_q), 500)

for idx, (model_name, color) in enumerate(zip(results.keys(), colors)):
    if idx < len(axes):
        ax = axes[idx]
        result = results[model_name]
        
        # Plot data and fit
        ax.errorbar(length_q, q_factor, yerr=q_err, xerr=length_q_err, 
                    fmt="o", label="Data", color="green", markersize=8, capsize=4)
        
        curve = result['func'](xs_q, *result['popt'])
        label_text = result['label_template'].format(*result['popt'])
        ax.plot(xs_q, curve, color=color, linewidth=2.5, label=f"{model_name}")
        
        # Add statistics
        stats_text = f"χ²ᵣ = {result['reduced_chi_squared']:.2f}\nR² = {result['r_squared']:.4f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel("Length (m)", fontsize=12)
        ax.set_ylabel("Q-factor", fontsize=12)
        ax.set_title(f"{model_name} Model", fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

# Hide unused subplots
for idx in range(len(results), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("q_factor_model_comparison.png", dpi=300)
print("Saved: q_factor_model_comparison.png")
plt.show()

# Print summary
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(f"{'Model':<12} {'χ²':<12} {'χ²ᵣ':<12} {'R²':<12} {'Parameters'}")
print("-"*70)
for model_name, result in results.items():
    params_str = ", ".join([f"{p:.3f}" for p in result['popt']])
    print(f"{model_name:<12} {result['chi_squared']:<12.2f} "
          f"{result['reduced_chi_squared']:<12.2f} {result['r_squared']:<12.4f} {params_str}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)

# Find best model by different criteria
best_r2 = max(results.items(), key=lambda x: x[1]['r_squared'])
best_chi = min(results.items(), key=lambda x: abs(x[1]['reduced_chi_squared'] - 1.0))

print(f"Best model by R² (highest): {best_r2[0]} with R² = {best_r2[1]['r_squared']:.4f}")
print(f"Best model by χ²ᵣ (closest to 1): {best_chi[0]} with χ²ᵣ = {best_chi[1]['reduced_chi_squared']:.2f}")
print()
print("All models have high reduced χ² (>> 1), suggesting:")
print("  1. Uncertainties may be underestimated")
print("  2. Real physical variation not captured by simple models")
print("  3. Possible systematic effects")
print()
print(f"The {best_r2[0]} model has the highest R² ({best_r2[1]['r_squared']:.4f}),")
print(f"explaining ~{best_r2[1]['r_squared']*100:.1f}% of variance.")
print("However, with such high χ²ᵣ, none of the models are statistically good fits.")
print()
print("Recommendation: Q-factor may not have a simple systematic relationship")
print("with length, or additional variables affect the measurement.")
