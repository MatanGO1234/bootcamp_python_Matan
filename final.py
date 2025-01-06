# ===================== Import ===============================
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import t
from scipy.stats import norm
from ipywidgets import IntRangeSlider, interact, FloatSlider
from statsmodels.stats.multitest import fdrcorrection

# =================== Parameters ================================
sampling_rate = 30000  # Hz
refractory_period = 0.001  # 1 ms
min_distance = int(refractory_period * sampling_rate)
alpha = 0.05


# =================== Step 1: Data pre-processing =====================
# Function for filtering
def band_pass_filter(signal, lowcut, highcut, sampling_rate=30000):
    nyquist = 0.5 * sampling_rate
    b, a = butter(2, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)


# Upload the data and filter
signal = np.load("C:\\test1\\matan_bootcamp_python\\bic13-ch259.npy")[:30000]
filtered_signal = band_pass_filter(signal, 300, 3000, sampling_rate)


# ============================ Step 2: FDR Correction ====================================
# Apply FDR correction
def apply_fdr(filtered_signal, alpha):
    p_values = 2 * (1 - norm.cdf(np.abs(filtered_signal / np.std(filtered_signal))))
    print(f"Min p-value: {np.min(p_values)}, Max p-value: {np.max(p_values)}")
    spikes_indices, _ = fdrcorrection(p_values, alpha=alpha)
    return spikes_indices


# Apply FDR correction
spike_indices = apply_fdr(filtered_signal, alpha)
# spike_indices = np.array(spike_indices, dtype=bool)


# Calculate FDR thresholds
def Calculate_FDR_thresholds(filtered_signal, spike_indices):
    if np.any((filtered_signal > 0) & spike_indices):
        positive_fdr = np.min(filtered_signal[(filtered_signal > 0) & spike_indices])
    else:
        positive_fdr = np.nan

    if np.any((filtered_signal < 0) & spike_indices):
        negative_fdr = np.max(filtered_signal[(filtered_signal < 0) & spike_indices])
    else:
        negative_fdr = np.nan
    return positive_fdr, negative_fdr


positive_fdr, negative_fdr = Calculate_FDR_thresholds(filtered_signal, spike_indices)


# =============================== Step 3: Spike Detection =========================================


# enforce_refractory for spikes_indices
def enforce_refractory(spike_indices, min_distance):
    final_spikes = []
    last_spike = -np.inf
    for spike in spike_indices:
        if spike - last_spike > min_distance:
            final_spikes.append
            last_spike = spike
    return np.array(final_spikes)


positive_indices = np.where((filtered_signal >= positive_fdr) & (filtered_signal > 0) & spike_indices)[0]
negative_indices = np.where((filtered_signal <= negative_fdr) & (filtered_signal < 0) & spike_indices)[0]

positive_spikes = enforce_refractory(positive_indices, min_distance)
negetive_spikes = enforce_refractory(negative_indices, min_distance)


# ================================ Step 4: Spike characterization ========================================


# Calculate spike peak and duration
def calculate_spike_characteristics(signal, spike_indices, threshold, mode="positive"):
    spike_characteristics = []
    for spike in spike_indices:
        if mode == "positive":
            start_spike = spike
            while start_spike > 0 and signal[start_spike] > threshold:
                start_spike -= 1

            end_spike = spike
            while end_spike < len(signal) and signal[end_spike] > threshold:
                end_spike -= 1

        else:  # negetive
            start_spike = spike
            while start_spike > 0 and signal[start_spike] < threshold:
                start_spike -= 1
            end_spike = spike
            while end_spike < len(signal) and signal[end_spike] < threshold:
                end_spike -= 1
        if start_spike >= end_spike:
            continue

        # finding the peak of the spike
        if mode == "negetive":
            local_peak_spike = np.argmax(signal[start_spike:end_spike]) + start_spike
        else:
            local_peak_spike = np.argmin(signal[start_spike:end_spike]) + start_spike
        spike_characteristics.append(
            {
                "peak": signal[local_peak_spike],
                "duration": end_spike - start_spike,
                "start_index": start_spike,
                "end_index": end_spike,
                "peak_index": local_peak_spike,
            }
        )
    return spike_characteristics


positive_spike_char = calculate_spike_characteristics(
    filtered_signal, positive_spikes, threshold=positive_fdr, mode="positive"
)
negative_spike_char = calculate_spike_characteristics(
    filtered_signal, negetive_spikes, threshold=negative_fdr, mode="negative"
)


# ============================= Step 5: Visualzation ===========================================
# plot signal with FDR correction spike markers
def plot_signal_with_spikes(signal, positive_spike_char, negative_spike_char, positive_fdr, negative_fdr):
    def plot(x_range):
        x_min, x_max = x_range
        plt.figure(figsize=(12, 6))
        plt.plot(signal, color="lightblue", label="Filtered Signal")

        if not np.isnan(positive_fdr):
            plt.axhline(positive_fdr, color="green", linestyle="--", label="Positive FDR Threshold")
        if not np.isnan(negative_fdr):
            plt.axhline(negative_fdr, color="green", linestyle="--", label="Negative FDR Threshold")

        for spike in positive_spike_cha:
            plt.scatter(
                spike["peak_index"],
                spike["peak"],
                color="red",
                label="Positive Spike" if "Positive Spike" not in plt.gca().get_legend_handles_labels()[1] else "",
            )
        for spike in negative_spike_char:
            plt.scatter(
                spike["peak_index"],
                spike["peak"],
                color="blue",
                label="Negative Spike" if "Negative Spike" not in plt.gca().get_legend_handles_labels()[1] else "",
            )

        plt.xlim(x_min, x_max)
        plt.title("Interactive Signal with Detected Spikes")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude (µV)")
        plt.legend(loc="lower left", fontsize=8)
        plt.grid()
        plt.tight_layout()
        plt.show()
        # Create IntRangeSlider for zoom

    x_slider = IntRangeSlider(
        value=[0, len(signal)],
        min=0,
        max=len(signal),
        step=100,
        description="X Range:",
        continuous_update=False,
    )

    # Connect slider to plot function
    interact(plot, x_range=x_slider)


# plot problemtic signals close to FDR threshholds
def plot_problematic_signals(signal, spike_indices, positive_fdr, negative_fdr):
    problematic_indices = np.where((~spike_indices) & ((signal > positive_fdr * 0.9) | (signal < negative_fdr * 1.1)))[
        0
    ]
    plt.figure(figsize=(12, 6))
    plt.plot(signal, color="lightblue", label="Filtered Signal")
    plt.scatter(problematic_indices, signal[problematic_indices], color="orange", label="Problematic Signals")

    if not np.isnan(positive_fdr):
        plt.axhline(positive_fdr, color="green", linestyle="--", label="Positive FDR Threshold")
    if not np.isnan(negative_fdr):
        plt.axhline(negative_fdr, color="green", linestyle="--", label="Negative FDR Threshold")

    plt.title("Signals Close to FDR Threshold")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude (µV)")
    plt.legend(loc="best", fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()


# Plot results
plot_signal_with_spikes(filtered_signal, positive_spike_char, negative_spike_char, positive_fdr, negative_fdr)
# Plot problematic signals close to FDR threshold
plot_problematic_signals(filtered_signal, spike_indices, positive_fdr, negative_fdr)
