import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

data_dir = 'Y:/zhu-lab/Stefano/Calcium TGT/BCR TGT GCaMP-TdT/photobleaching/results/'

# Load the CSV file (adjust the filename/path as needed)
df = pd.read_csv(os.path.join(data_dir, "R_IgG_ITT2F_BSA-NIP_photobleaching_013_tracks_mfi_data.csv"))

# Only keep frames >= 0 (ignoring any negative frames)
df = df[df['frame'] >= 0]

# Extract time (assuming data is recorded at 1 fps)
t = df['frame'].values

# Extract the mean values and SEM for GCaMP and RFP channels.
y_gcamp = df['GCaMP_mean'].values
yerr_gcamp = df['GCaMP_sem'].values

y_rfp = df['RFP_mean'].values
yerr_rfp = df['RFP_sem'].values

# Simplified single exponential decay model: f(t) = exp(-k*t)
def single_exp(t, k):
    return np.exp(-k * t)

# Simplified double exponential decay model:
# f(t) = A*exp(-k1*t) + (1-A)*exp(-k2*t)
def double_exp(t, A, k1, k2):
    return A * np.exp(-k1 * t) + (1 - A) * np.exp(-k2 * t)

# Initial parameter guesses
p0_single = [0.1]
p0_double = [0.5, 0.2, 0.01]

# --- Fitting for GCaMP (fitting the mean only) ---
popt_gcamp_single, pcov_gcamp_single = curve_fit(
    single_exp, t, y_gcamp, p0=p0_single, maxfev=10000
)
popt_gcamp_double, pcov_gcamp_double = curve_fit(
    double_exp, t, y_gcamp, p0=p0_double, maxfev=10000
)

# --- Fitting for RFP (fitting the mean only) ---
popt_rfp_single, pcov_rfp_single = curve_fit(
    single_exp, t, y_rfp, p0=p0_single, maxfev=10000
)
popt_rfp_double, pcov_rfp_double = curve_fit(
    double_exp, t, y_rfp, p0=p0_double, maxfev=10000
)

# Calculate parameter standard errors from the covariance matrices
perr_gcamp_single = np.sqrt(np.diag(pcov_gcamp_single))
perr_gcamp_double = np.sqrt(np.diag(pcov_gcamp_double))
perr_rfp_single   = np.sqrt(np.diag(pcov_rfp_single))
perr_rfp_double   = np.sqrt(np.diag(pcov_rfp_double))

# Generate a dense time vector for plotting the fitted curves
t_fit = np.linspace(t.min(), t.max(), 200)
gcamp_single_fit = single_exp(t_fit, *popt_gcamp_single)
gcamp_double_fit = double_exp(t_fit, *popt_gcamp_double)
rfp_single_fit   = single_exp(t_fit, *popt_rfp_single)
rfp_double_fit   = double_exp(t_fit, *popt_rfp_double)

# Plot the fits for GCaMP
plt.figure(figsize=(10, 5))
plt.errorbar(t, y_gcamp, yerr=yerr_gcamp, fmt='o', label='GCaMP data', capsize=3)
plt.plot(t_fit, gcamp_single_fit, '-', label='Single exponential fit', linewidth=2)
plt.plot(t_fit, gcamp_double_fit, '--', label='Double exponential fit', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('GCaMP MFI (normalized)')
plt.title('Photobleaching Fit for GCaMP')
plt.legend()
plt.tight_layout()
plt.savefig("GCaMP_fit.png", dpi=300)
plt.show()

# Plot the fits for RFP
plt.figure(figsize=(10, 5))
plt.errorbar(t, y_rfp, yerr=yerr_rfp, fmt='o', label='RFP data', capsize=3)
plt.plot(t_fit, rfp_single_fit, '-', label='Single exponential fit', linewidth=2)
plt.plot(t_fit, rfp_double_fit, '--', label='Double exponential fit', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('RFP MFI (normalized)')
plt.title('Photobleaching Fit for RFP')
plt.legend()
plt.tight_layout()
plt.savefig("RFP_fit.png", dpi=300)
plt.show()

# Save the fitted parameters to a CSV file.
rows = []

# For GCaMP single exponential fit:
params_names_single = ['k']
for name, val, err in zip(params_names_single, popt_gcamp_single, perr_gcamp_single):
    rows.append({"Channel": "GCaMP", "Model": "Single Exponential", "Parameter": name, "Value": val, "StdErr": err})

# For GCaMP double exponential fit:
params_names_double = ['A', 'k1', 'k2']
for name, val, err in zip(params_names_double, popt_gcamp_double, perr_gcamp_double):
    rows.append({"Channel": "GCaMP", "Model": "Double Exponential", "Parameter": name, "Value": val, "StdErr": err})

# For RFP single exponential fit:
for name, val, err in zip(params_names_single, popt_rfp_single, perr_rfp_single):
    rows.append({"Channel": "RFP", "Model": "Single Exponential", "Parameter": name, "Value": val, "StdErr": err})

# For RFP double exponential fit:
for name, val, err in zip(params_names_double, popt_rfp_double, perr_rfp_double):
    rows.append({"Channel": "RFP", "Model": "Double Exponential", "Parameter": name, "Value": val, "StdErr": err})

fit_params_df = pd.DataFrame(rows)
fit_params_df.to_csv(os.path.join(data_dir, "fitted_parameters.csv"), index=False)

print("Fitting complete. Fitted parameters saved to 'fitted_parameters.csv'.")
