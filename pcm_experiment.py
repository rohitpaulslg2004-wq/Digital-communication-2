"""
Experiment 4 - Uniform Quantization and PCM
=============================================
Implements a uniform PCM quantizer from first principles.

Pipeline:
  1. Normalize a sinusoid to [-1, 1].
  2. For n = 2,3,4,6,8 bits (L = 2^n levels):
       - Quantize the signal (mid-rise uniform quantizer)
       - Generate quantizer indices (0 .. L-1)
       - Generate PCM binary codewords (n-bit strings)
       - Compute quantization error, MSE, and SQNR
  3. Compare measured SQNR against the theoretical full-scale
     sinusoid approximation:  SQNR_dB = 6.02*n + 1.76
  4. Produce required visualizations.
  5. Run mandatory validation checks.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. Signal generation and normalization
# ---------------------------------------------------------------
Fs = 8000          # sampling frequency (Hz)
f0 = 200           # tone frequency (Hz)
T = 0.02           # duration (s) -> 4 full cycles at 200 Hz
t = np.arange(0, T, 1 / Fs)

# Raw sinusoid with an arbitrary (non full-scale) amplitude/offset,
# to make the normalization step meaningful.
raw = 0.83 * np.sin(2 * np.pi * f0 * t) + 0.05

def normalize_to_unit(x):
    """Normalize a signal to occupy exactly [-1, 1] (full-scale)."""
    peak = np.max(np.abs(x))
    return x / peak

x = normalize_to_unit(raw)
print(f"Normalized signal range: [{x.min():.4f}, {x.max():.4f}]")

# ---------------------------------------------------------------
# 2. Uniform PCM quantizer (mid-rise), first principles
# ---------------------------------------------------------------
def uniform_pcm_quantize(x, n_bits, full_scale=1.0):
    """
    Mid-rise uniform quantizer over [-full_scale, full_scale].

    Returns
    -------
    xq      : quantized (reconstructed) samples
    idx     : quantizer indices, integers in [0, L-1]
    words   : list of n-bit binary strings (PCM codewords)
    delta   : step size
    """
    L = 2 ** n_bits
    delta = (2 * full_scale) / L

    # Clip to full-scale range to avoid overload / out-of-range indices
    x_clipped = np.clip(x, -full_scale, full_scale - 1e-12)

    # Mid-rise quantizer: index = floor((x + full_scale) / delta)
    idx = np.floor((x_clipped + full_scale) / delta).astype(int)
    idx = np.clip(idx, 0, L - 1)   # safety clamp -> guarantees 0..L-1

    # Reconstruction (decoder) level = center of the idx-th interval
    xq = -full_scale + delta * (idx + 0.5)

    # PCM codewords: natural binary, n bits, MSB first
    words = [format(i, f'0{n_bits}b') for i in idx]

    return xq, idx, words, delta


def measure_performance(x, xq):
    """Quantization error, MSE and SQNR (dB)."""
    err = x - xq
    mse = np.mean(err ** 2)
    signal_power = np.mean(x ** 2)
    sqnr_db = 10 * np.log10(signal_power / mse) if mse > 0 else np.inf
    return err, mse, sqnr_db


def theoretical_sqnr(n_bits):
    """Standard full-scale sinusoid approximation."""
    return 6.02 * n_bits + 1.76


# ---------------------------------------------------------------
# 3. Run the experiment for n = 2,3,4,6,8 bits
# ---------------------------------------------------------------
bit_depths = [2, 3, 4, 6, 8]
results = {}

for n in bit_depths:
    xq, idx, words, delta = uniform_pcm_quantize(x, n)
    err, mse, sqnr_meas = measure_performance(x, xq)
    sqnr_theo = theoretical_sqnr(n)
    results[n] = dict(xq=xq, idx=idx, words=words, delta=delta,
                       err=err, mse=mse, sqnr_meas=sqnr_meas,
                       sqnr_theo=sqnr_theo)

# ---------------------------------------------------------------
# 4. Mandatory validation
# ---------------------------------------------------------------
print("\n--- Mandatory validation ---")
all_ok = True
for n in bit_depths:
    r = results[n]
    L = 2 ** n
    idx_ok = np.all((r['idx'] >= 0) & (r['idx'] <= L - 1))
    word_len_ok = all(len(w) == n for w in r['words'])
    status = "PASS" if (idx_ok and word_len_ok) else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"n={n:2d} bits | L={L:3d} | indices in [0,{L-1}]: {idx_ok} "
          f"| PCM word length == n: {word_len_ok} | {status}")

print(f"\nOverall validation: {'ALL CHECKS PASSED' if all_ok else 'FAILURE DETECTED'}")

# ---------------------------------------------------------------
# 5. Numerical summary table
# ---------------------------------------------------------------
print("\n--- SQNR summary ---")
print(f"{'n (bits)':>8} | {'L':>5} | {'MSE':>12} | {'SQNR meas (dB)':>15} | "
      f"{'SQNR theory (dB)':>17} | {'Diff (dB)':>10}")
for n in bit_depths:
    r = results[n]
    diff = r['sqnr_meas'] - r['sqnr_theo']
    print(f"{n:8d} | {2**n:5d} | {r['mse']:12.3e} | {r['sqnr_meas']:15.3f} | "
          f"{r['sqnr_theo']:17.3f} | {diff:10.3f}")

# Sample PCM words printout for n=4 (first 10 samples) as a concrete example
print("\nSample PCM codewords (n=4 bits), first 10 samples:")
for i in range(10):
    print(f"  sample {i:2d}: x={x[i]:+.4f} -> idx={results[4]['idx'][i]:2d} "
          f"-> word={results[4]['words'][i]}")

# =================================================================
# REQUIRED VISUALIZATIONS
# =================================================================
plt.rcParams.update({'font.size': 10})

# ---- (a) Original / quantized waveform (for a representative set of n) ----
fig, axes = plt.subplots(len(bit_depths), 1, figsize=(9, 12), sharex=True)
for ax, n in zip(axes, bit_depths):
    r = results[n]
    ax.plot(t * 1e3, x, 'k-', lw=1.2, label='Original (normalized)')
    ax.step(t * 1e3, r['xq'], where='mid', color='crimson', lw=1.0,
             label=f'Quantized (n={n} bits)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'n = {n} bits, L = {2**n} levels, SQNR = {r["sqnr_meas"]:.2f} dB')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('Time (ms)')
fig.suptitle('Original vs Quantized Waveform', fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig('/home/claude/pcm/01_waveform_original_vs_quantized.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---- (b) Staircase characteristic (input-output transfer curve) ----
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()
x_sweep = np.linspace(-1, 1, 2000)
for ax, n in zip(axes, bit_depths):
    xq_sweep, _, _, delta = uniform_pcm_quantize(x_sweep, n)
    ax.plot(x_sweep, x_sweep, 'k--', lw=0.8, label='Ideal (no quant.)')
    ax.plot(x_sweep, xq_sweep, color='steelblue', lw=1.3, label='Quantizer output')
    ax.set_title(f'n = {n} bits (Δ = {delta:.4f})')
    ax.set_xlabel('Input x')
    ax.set_ylabel('Output $x_q$')
    ax.grid(alpha=0.3)
    if n == bit_depths[0]:
        ax.legend(fontsize=8)
axes[-1].axis('off')
fig.suptitle('Staircase (Input-Output) Characteristic of the Uniform Quantizer', fontsize=13)
fig.tight_layout()
fig.savefig('/home/claude/pcm/02_staircase_characteristic.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---- (c) Error waveform / histogram ----
fig, axes = plt.subplots(len(bit_depths), 2, figsize=(11, 12))
for row, n in enumerate(bit_depths):
    r = results[n]
    ax_wave = axes[row, 0]
    ax_hist = axes[row, 1]

    ax_wave.plot(t * 1e3, r['err'], color='darkorange', lw=1.0)
    ax_wave.axhline(r['delta'] / 2, color='gray', ls='--', lw=0.8)
    ax_wave.axhline(-r['delta'] / 2, color='gray', ls='--', lw=0.8)
    ax_wave.set_title(f'Error waveform, n={n} bits')
    ax_wave.set_ylabel('Error')
    ax_wave.grid(alpha=0.3)

    ax_hist.hist(r['err'], bins=30, color='seagreen', alpha=0.8)
    ax_hist.set_title(f'Error histogram, n={n} bits')
    ax_hist.set_xlabel('Error amplitude')
    ax_hist.grid(alpha=0.3)

axes[-1, 0].set_xlabel('Time (ms)')
fig.suptitle('Quantization Error: Waveform and Histogram', fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig('/home/claude/pcm/03_error_waveform_histogram.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ---- (d) SQNR vs bits ----
n_arr = np.array(bit_depths)
sqnr_meas_arr = np.array([results[n]['sqnr_meas'] for n in bit_depths])
sqnr_theo_arr = np.array([results[n]['sqnr_theo'] for n in bit_depths])

# Also compute a finer theoretical curve for a smooth reference line
n_fine = np.linspace(1, 9, 100)
sqnr_theo_fine = theoretical_sqnr(n_fine)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n_fine, sqnr_theo_fine, 'k--', lw=1.2, label='Theory: 6.02n + 1.76 dB')
ax.plot(n_arr, sqnr_meas_arr, 'o-', color='crimson', ms=7, lw=1.5, label='Measured (simulation)')
for n_i, s_i in zip(n_arr, sqnr_meas_arr):
    ax.annotate(f'{s_i:.1f} dB', (n_i, s_i), textcoords="offset points",
                xytext=(0, 8), ha='center', fontsize=8)
ax.set_xlabel('Number of bits (n)')
ax.set_ylabel('SQNR (dB)')
ax.set_title('SQNR vs. Number of Bits')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('/home/claude/pcm/04_sqnr_vs_bits.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print("\nAll figures saved to /home/claude/pcm/")
