import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from ipywidgets import interact, FloatSlider, IntSlider, IntRangeSlider
from scipy.stats import t
from scipy.stats import norm
from ipywidgets import interact
from statsmodels.stats.multitest import fdrcorrection

# Parameters
sampling_rate = 30000  # Hz
refractory_period = 0.002  # 1 ms
min_distance = int(refractory_period * sampling_rate)
alpha = 0.05

# ===================Step 1: Data Preprocessing====================
# Function for filtering
def band_pass_filter(signal, lowcut, highcut, sampling_rate=30000):
    nyquist = 0.5 * sampling_rate
    b, a = butter(2, [lowcut / nyquist, highcut / nyquist], btype="band")
    return filtfilt(b, a, signal)

# Upload the data and filter
signal = np.load("C:\\test1\\matan_bootcamp_python\\bic13-ch259.npy")[:30000]
filtered_signal = band_pass_filter(signal, 300, 10000, sampling_rate)

# ========================Step 2: FDR Correction========================
# FDR correction
def apply_fdr(filtered_signal, alpha):
    p_values = 2 * (1 - norm.cdf(np.abs(filtered_signal / np.std(filtered_signal))))
    spikes_indices, _ = fdrcorrection(p_values, alpha=alpha)
    return spikes_indices

# Apply FDR & spikes_idx to values & significant idx
spikes_indices = apply_fdr(filtered_signal, alpha)
spikes_values = filtered_signal[spikes_indices]
indices = np.where(spikes_indices)[0]


# Calculate FDR thresholds
def Calculate_FDR_thresholds(spikes_values):
    if np.any(spikes_values > 0):
        positive_fdr = np.min(spikes_values[spikes_values > 0])
    else:
        positive_fdr = np.nan

    if np.any(spikes_values < 0):
        negative_fdr = np.max(spikes_values[spikes_values < 0])
    else:
        negative_fdr = np.nan
    return positive_fdr, negative_fdr

# Apply FDR thresholds
positive_fdr, negative_fdr = Calculate_FDR_thresholds(spikes_values)

#=================Step 3: Spike Detection====================
# enforce_refractory for spikes_values
def enforce_refractory(spikes_indices, min_distance):
    final_spikes = []
    last_spike = -np.inf
    for spikes in spikes_indices:
        if spikes - last_spike > min_distance:
            final_spikes.append(spikes)
            last_spike = spikes
    return np.array(final_spikes)

 # sorting to signifiant pos and neg
positive_indices_idx = np.where((filtered_signal >= positive_fdr) & (filtered_signal > 0))[0]
negative_indices_idx = np.where((filtered_signal <= negative_fdr) & (filtered_signal < 0))[0]   

# apply enforce_refractory
positive_spikes_idx = enforce_refractory(positive_indices_idx, min_distance)
negative_spikes_idx = enforce_refractory(negative_indices_idx, min_distance)

# ============================Step 4: Spike Characterization=======================
# Calculate spike peak and duration
def calculate_spike_characteristics(signal, spikes_indices, positive_fdr, negative_fdr, mode="positive"):
    spike_characteristics = []
    buffer_size = 50  # Adding buffer size for optimization
    for spike_idx in spikes_indices:  # spike_idx is an global index in signal
        start_spike = max(0, spike_idx - buffer_size)
        end_spike = min(len(filtered_signal), spike_idx + buffer_size)
        if mode == "positive":
            while start_spike > 0 and filtered_signal[start_spike] > positive_fdr:
                start_spike -= 1

            while end_spike < len(filtered_signal) and filtered_signal[end_spike] > positive_fdr:
                end_spike += 1

        else:  # negative
            while start_spike > 0 and filtered_signal[start_spike] < negative_fdr:
                start_spike -= 1

            while end_spike < len(filtered_signal) and filtered_signal[end_spike] < negative_fdr:
                end_spike += 1

        if start_spike >= end_spike:
            continue

        # finding the peak or the trough of the spikes
        if mode == "negative":
            local_peak_spike = np.argmin(filtered_signal[start_spike:end_spike]) + start_spike
        else:
            local_peak_spike = np.argmax(filtered_signal[start_spike:end_spike]) + start_spike

        # local_peak_spike is relatiivty index compare to filtered_signal
        peak_index = local_peak_spike
        peak_value = filtered_signal[peak_index]
        spike_characteristics.append(
            {
                "peak_index": peak_index,
                "peak": peak_value,
                "duration": end_spike - start_spike,
                "start_index": start_spike,
                "end_index": end_spike,
            }
        )
    return spike_characteristics

# apply def location spikes -- peak & trough
positive_spike_char = calculate_spike_characteristics(
    filtered_signal, positive_spikes_idx, positive_fdr, negative_fdr, mode="positive"
)
negative_spike_char = calculate_spike_characteristics(
    filtered_signal, negative_spikes_idx, positive_fdr, negative_fdr, mode="negative"
)

# Filtering repted pos spikes 
unique_positive_spikes = list({spike["peak_index"]: spike for spike in positive_spike_char}.values())
# Filtering repted neg spikes 
unique_negative_spikes = list({spike["peak_index"]: spike for spike in negative_spike_char}.values())

# =============================Step 5: Visualization===============================
## plot signal with FDR correction spikes markers
def plot_signal_with_spikes(filtered_signal, unique_positive_spikes, unique_negative_spikes, positive_fdr, negative_fdr):
    def plot(x_range):
        x_min, x_max = x_range
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_signal, color="lightblue", label="Filtered Signal")

        if not np.isnan(positive_fdr):
            plt.axhline(positive_fdr, color="green", linestyle="--", label="Positive FDR Threshold")
        if not np.isnan(negative_fdr):
            plt.axhline(negative_fdr, color="purple", linestyle="--", label="Negative FDR Threshold")

        # Positive spikes
        plt.scatter(
            [spike["peak_index"] for spike in unique_positive_spikes],
            [spike["peak"] for spike in unique_positive_spikes],
            color="red",
            label="Positive Spike" if "Positive Spike" not in plt.gca().get_legend_handles_labels()[1] else "",
        )

        # Negative spikes
        plt.scatter(
            [spike["peak_index"] for spike in unique_negative_spikes],
            [spike["peak"] for spike in unique_negative_spikes],
            color="blue",
            label="Negative Spike" if "Negative Spike" not in plt.gca().get_legend_handles_labels()[1] else "",
        )

        plt.xlim(0, len(filtered_signal))
        plt.title("Interactive Signal with Detected Spikes")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude (µV)")
        plt.legend(loc="lower left", fontsize=8)
        plt.grid()
        plt.tight_layout()
        plt.show()

    x_slider = IntRangeSlider(
        value=[0, len(filtered_signal)],
        min=0,
        max=len(filtered_signal),
        step=100,
        description="X Range:",
        continuous_update=False,
    )

    interact(plot, x_range=x_slider)

