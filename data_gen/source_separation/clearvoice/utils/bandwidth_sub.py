import numpy as np
import soundfile as sf
import librosa
import os
from scipy.signal import butter, filtfilt, stft, istft

# Step 1: Load audio files
def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=48000)
    #audio, fs = sf.read(audio_path)
    return audio, sr

# Step 2: Detect effective signal bandwidth
def detect_bandwidth_org(signal, fs, energy_threshold=0.95):
    f, t, Zxx = stft(signal, fs=fs)
    psd = np.abs(Zxx)**2
    total_energy = np.sum(psd)
    cumulative_energy = np.cumsum(np.sum(psd, axis=1)) / total_energy
    f_low = f[np.argmax(cumulative_energy > (1 - energy_threshold))]
    f_high = f[np.argmax(cumulative_energy >= energy_threshold)]
    return f_low, f_high

def detect_bandwidth(signal, fs, energy_threshold=0.9996):
    f, t, Zxx = stft(signal, fs=fs)
    psd = np.abs(Zxx)**2
    total_energy = np.sum(psd)
    cumulative_energy = np.cumsum(np.sum(psd, axis=1)) / total_energy
    
    # Exclude DC component (0 Hz)
    valid_indices = np.where(f > 0)[0]
    f_low = f[valid_indices][np.argmax(cumulative_energy[valid_indices] > (1 - energy_threshold))]
    f_high = f[np.argmax(cumulative_energy >= energy_threshold)]
    return f_low, f_high
    
# Step 3: Apply bandpass and lowpass filters
def bandpass_filter(signal, fs, f_low, f_high):
    nyquist = 0.5 * fs
    low = f_low / nyquist
    high = f_high / nyquist
    b, a = butter(N=4, Wn=[low, high], btype='band')
    return filtfilt(b, a, signal)

def lowpass_filter(signal, fs, cutoff):
    nyquist = 0.5 * fs
    cutoff_normalized = cutoff / nyquist
    b, a = butter(N=4, Wn=cutoff_normalized, btype='low')
    return filtfilt(b, a, signal)

def highpass_filter(signal, fs, cutoff):
    nyquist = 0.5 * fs
    cutoff_normalized = cutoff / nyquist
    b, a = butter(N=4, Wn=cutoff_normalized, btype='high')
    return filtfilt(b, a, signal)

# Step 4: Replace bandwidth
def replace_bandwidth(signal1, signal2, fs, f_low, f_high):
    # Extract effective band from signal1
    #effective_band = bandpass_filter(signal1, fs, f_low, f_high)
    effective_band = lowpass_filter(signal1, fs, f_high)
    # Extract lowpass band from signal2
    #signal2_lowpass = lowpass_filter(signal2, fs, f_high)
    signal2_highpass = highpass_filter(signal2, fs, f_high)
    
    # Match lengths of the two signals
    min_length = min(len(effective_band), len(signal2_highpass))
    effective_band = effective_band[:min_length]
    signal2_highpass = signal2_highpass[:min_length]
    
    # Combine the two signals
    return signal2_highpass + effective_band

# Step 5: Smooth transitions
def smooth_transition(signal1, signal2, fs, transition_band=100):
    fade = np.linspace(0, 1, int(transition_band * fs / 1000))
    crossfade = np.concatenate([fade, np.ones(len(signal1) - len(fade))])
    min_length = min(len(signal1), len(signal2))
    smoothed_signal = (1 - crossfade) * signal2[:min_length] + crossfade * signal1[:min_length]
    return smoothed_signal

# Step 6: Save audio
def save_audio(file_path, audio, fs):
    sf.write(file_path, audio, fs)


def bandwidth_sub(low_bandwidth_audio, high_bandwidth_audio, fs=48000):
    # Detect effective bandwidth of the first signal
    f_low, f_high = detect_bandwidth(low_bandwidth_audio, fs)
        
    # Replace the lower frequency of the second audio
    substituted_audio = replace_bandwidth(low_bandwidth_audio, high_bandwidth_audio, fs, f_low, f_high)

    # Optional: Smooth the transition
    smoothed_audio = smooth_transition(substituted_audio, low_bandwidth_audio, fs)
    return smoothed_audio
        
# Main process
if __name__ == "__main__":
    low_spectra_dir = 'LJSpeech_22k'
    upper_spectra_dir = 'LJSpeech_22k_hifi-sr_speech_g_03925000'
    output_dir = upper_spectra_dir+'_restored'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    filelist = [file for file in os.listdir(low_spectra_dir) if file.endswith('.wav')]
    for audio_name in filelist:
        audio1, fs1 = load_audio(low_spectra_dir + "/" + audio_name)  # Source for effective bandwidth
        audio2, fs2 = load_audio(upper_spectra_dir + "/" + audio_name.replace('.wav', '_generated.wav'))  # Target audio to replace lower frequencies

        if fs1 != 48000 or fs2 != 48000:
            raise ValueError("Both audio files must have a sampling rate of 48 kHz.")

        # Detect effective bandwidth of the first signal
        f_low, f_high = detect_bandwidth(audio1, fs1)

        # Replace the lower frequency of the second audio
        replaced_audio = replace_bandwidth(audio1, audio2, fs2, f_low, f_high)

        # Optional: Smooth the transition
        smoothed_audio = smooth_transition(replaced_audio, audio1, fs1)

        # Save the result
        save_audio(output_dir+"/"+audio_name, smoothed_audio, fs2)
