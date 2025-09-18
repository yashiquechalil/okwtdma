import numpy as np
from PIL import Image


processing_log = []


def reverse(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Reverse order of frames"""
    processing_log.append("Reverse order of frames")
    return np.flip(audio_data.reshape(-1, frame_size), axis=0)


def flip(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Reverse data within each frame"""
    audio_data = audio_data.reshape(-1, frame_size)
    processing_log.append("Reverse data within each frame")
    return np.flip(audio_data, axis=1)


def invert_phase(
    audio_data: np.ndarray,
    frame_size: int,
) -> np.ndarray:
    """Invert phase"""
    audio_data = audio_data.reshape(-1, frame_size)
    processing_log.append("Invert phase")
    return audio_data * -1


def shuffle(
    audio_data: np.ndarray, frame_size: int, num_groups: int, seed: int
) -> np.ndarray:
    """Randomize order of frames"""
    mutable_copy = np.copy(audio_data).reshape(-1)

    max_frames = audio_data.size // frame_size
    if (num_groups == 0) or (num_groups >= max_frames):
        num_groups = audio_data.size // frame_size

    num_frames = mutable_copy.size // (frame_size * num_groups) * num_groups

    mutable_copy = mutable_copy[: num_frames * frame_size].reshape(
        num_groups, -1
    )

    bit_generator = np.random.PCG64(seed=seed)
    rng = np.random.Generator(bit_generator)

    rng.shuffle(mutable_copy)
    mutable_copy = mutable_copy.reshape(-1)

    processing_log.append(f"Shuffle frames ({num_groups} groups)")
    return mutable_copy


def fade(
    audio_data: np.ndarray, frame_size: int, fades: str | list[str]
) -> np.ndarray:
    """Apply fade-in and fade-out to each frame"""

    audio_data_as_frames = audio_data.reshape(-1, frame_size)

    # Convert 'fades' values from str into int
    int_fades = []
    for value in fades:
        if "%" in value:
            int_fades.append(round(frame_size * int(value.strip("%")) / 100))
        else:
            int_fades.append(int(value))

    if len(int_fades) == 1:
        fade_in = fade_out = int_fades[0]
    elif len(int_fades) >= 2:
        fade_in, fade_out = int_fades[0], int_fades[1]

    mutable_copy = np.copy(audio_data_as_frames)

    for i in range(mutable_copy.shape[0]):
        if fade_in:
            mutable_copy[i, :fade_in] *= np.linspace(0.0, 1.0, fade_in)
        if fade_out:
            mutable_copy[i, -fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    processing_log.append(f"Apply fades to each frame: {fade_in}, {fade_out}")
    return mutable_copy


def sort(audio_data: np.ndarray, frame_size: int) -> np.ndarray:
    """Sort frames within wavetable"""
    sorted_frames = np.sort(audio_data.reshape(-1, frame_size), axis=0)
    processing_log.append("Sort frames")
    return sorted_frames


def trim(audio_data: np.ndarray, threshold) -> np.ndarray:
    """Remove values below threshold from the beginning and end of the array"""
    threshold = float(threshold)
    mask = np.abs(audio_data) < threshold
    start_idx = np.argmax(~mask)
    end_idx = len(audio_data) - np.argmax(~mask[::-1])

    processing_log.append(
        "Remove silence from the beginning and end of audio "
        f"(threshold: {threshold}, max value: {np.max(audio_data)})"
    )

    return audio_data[start_idx:end_idx]


def resize(
    audio_data: np.ndarray,
    target_num_frames: int,
    resize_mode: str,
    frame_size: int,
) -> np.ndarray:
    """Resize array to fit into targer_num_frames"""

    try:
        audio_data = audio_data.reshape(-1, frame_size)
    except Exception as e:
        raise NotImplementedError(f"Can't reshape {audio_data.shape}", e)

    resize_mode = resize_mode.lower()

    if len(resize_mode) < 3:
        raise SyntaxError(
            "Resize mode keyword must be more than 2 letters long"
        )

    if resize_mode in "truncate":
        # Simply trim the end
        processing_log.append("Resize mode: truncate")
        return audio_data[:target_num_frames]
    elif resize_mode in "linear":
        # Sample input array at equal intervals
        indices = np.linspace(
            0, len(audio_data) - 1, target_num_frames, dtype=int
        )
        processing_log.append("Resize mode: linear")
        return audio_data[indices]
    elif resize_mode in "bicubic":
        # Use Pillow library
        as_pil = Image.fromarray(audio_data)
        pil_resized = as_pil.resize(
            (frame_size, target_num_frames), Image.Resampling.BICUBIC
        )
        new_array = np.array(pil_resized, dtype=np.float32)
        reshaped = new_array.reshape(-1)
        processing_log.append("Resize mode: bicubic")
        return reshaped
    elif resize_mode in "geometric":
        # Sample more often at the beginning
        indices = np.geomspace(
            1,
            len(audio_data) - 1,
            target_num_frames,
            dtype=int,
        )
        processing_log.append("Resize mode: geometric")
        return audio_data.reshape(-1, frame_size)[indices]
    elif resize_mode in "percussive":
        # Sample all frames in first half of the sample, then sample linearly.
        first_half_size = target_num_frames // 2
        first_half = audio_data[:first_half_size]
        indices = np.linspace(
            0,
            len(audio_data[first_half_size:]) - 1,
            target_num_frames - first_half_size,
            dtype=int,
        )
        second_half = audio_data[first_half_size:][indices]
        new = np.concatenate((first_half, second_half), axis=0)
        processing_log.append("Resize mode: percussive")
        return new
    else:
        raise SyntaxError(f"Unknown keyword '{resize_mode}'")


def clip(audio_data: np.ndarray, clip_to: float):
    return np.clip(audio_data, -clip_to, clip_to)


def normalize(audio_data: np.ndarray, normalize_to: float) -> np.ndarray:
    """Apply peak normalization to a specified value (0.0 - 1.0)"""
    peak_value = np.abs(audio_data).max()
    ratio = normalize_to / peak_value
    normalized = audio_data * ratio
    processing_log.append("Apply peak normalization")
    return normalized


def maximize(
    audio_data: np.ndarray, frame_size: int, maximize_to: float
) -> np.ndarray:
    """Normalize each frame"""
    audio_data = audio_data.reshape(-1, frame_size)

    for frame in range(audio_data.shape[0]):
        audio_data[frame] = normalize(audio_data[frame], maximize_to)

    processing_log.append("Apply peak normalization to each frame")
    return audio_data


def interpolate(audio_data: np.ndarray, in_frame_size: int, out_frame_size):
    """Resample audio data to new frame size"""
    audio_frames = audio_data.reshape(-1, in_frame_size)
    target_size = audio_frames.shape[0] * out_frame_size
    linspace = np.linspace(0, len(audio_data), target_size)

    # TODO: fix edge values
    interpolated = np.interp(
        linspace, np.arange(len(audio_data)), audio_data
    ).astype(np.float32)

    peak_value = np.abs(interpolated).max()
    normalize_to = 1.0
    ratio = normalize_to / peak_value
    normalized = interpolated * ratio

    processing_log.append(
        f"Resample to new frame size: {in_frame_size} -> {out_frame_size}"
    )
    return normalized.astype(np.float32)


def overlap(audio_data: np.ndarray, frame_size: int, overlap_size: float):
    """Resize by overlapping frames"""

    raise NotImplementedError

    from scipy.signal import hann

    audio_data = audio_data.reshape(-1, frame_size)

    num_overlap_samples = int(overlap_size * audio_data.shape[1])

    fade_length = int(num_overlap_samples / 4)
    fade_in_window = hann(fade_length * 2)[:fade_length]
    fade_out_window = hann(fade_length * 2)[fade_length:]
    faded_array = audio_data.copy()

    for i in range(faded_array.shape[0]):
        faded_array[i, :fade_length] *= fade_in_window
        faded_array[i, -fade_length:] *= fade_out_window

    overlapping_array = np.concatenate(
        [
            faded_array[i, -num_overlap_samples * 2 :]
            for i in range(0, faded_array.shape[0])
        ],
        axis=0,
    )
    return overlapping_array


def splice(audio_data: np.ndarray, num_frames: int):
    raise NotImplementedError


def mix_with_reversed_copy(audio_data: np.ndarray, num_frames: int):
    raise NotImplementedError



def fundamental(audio, sr):
    """ Estimate f0 using FFT peak (Update to better algo's if needed)
    Args:
        audio (np.ndarray): Input audio signal.
        sr (int): Sample rate of the audio signal.
    Returns:
        float: Estimated fundamental frequency (f0) in Hz.
    """
    window = np.hamming(audio.size)
    sig = np.fft.fft(audio * window)
    freqs = np.fft.fftfreq(sig.size)
    i = np.argmax(np.abs(sig))
    f0 =  np.abs(freqs[i] * sr)
    processing_log.append(f"Median estimated fundamental frequency (f0): {np.nanmedian(f0)} Hz")
    return np.nanmedian(f0)

def track_f0(audio, sr, hop_length=256):
    
    """Estimate f0 over time using a simple autocorrelation method.
    Args:
        audio (np.ndarray): Input audio signal.
        sr (int): Sample rate of the audio signal.
        hop_length (int): Hop length between frames.        

    Returns:
        np.ndarray: Array of f0 estimates per frame (in Hz).    
    """
    frame_length = 2048
    num_frames = (len(audio) - frame_length) // hop_length + 1
    f0 = []
    for i in range(num_frames):
        frame = audio[i * hop_length : i * hop_length + frame_length]
        if len(frame) < frame_length:
            break
        # Remove DC
        frame = frame - np.mean(frame)
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')[frame_length-1:]
        # Find first minimum (ignore lag 0)
        d = np.diff(corr)
        start = np.where(d > 0)[0]
        if len(start) == 0:
            f0.append(np.nan)
            continue
        start = start[0]
        peak = np.argmax(corr[start:]) + start
        if peak == 0:
            f0.append(np.nan)
            continue
        f0_val = sr / peak
        f0.append(f0_val)
    f0 = np.array(f0)
    processing_log.append(f"Estimated tracked fundamental frequency (f0): {f0} Hz")
    return f0


def nearest_zero_crossing(audio, idx):
    """Find the nearest zero crossing to idx."""
    zero_crossings = np.where(np.diff(np.sign(audio)) > 0)[0]
    if len(zero_crossings) == 0:
        return idx  # fallback
    
    processing_log.append(f"Snapped to nearest zero crossing at sample {zero_crossings[np.argmin(np.abs(zero_crossings - idx))]}")
    return zero_crossings[np.argmin(np.abs(zero_crossings - idx))]


def slice_stretch(audio, frame_size, num_frames):
    """
    Evenly stretch/compress the file into L samples

    Args:
        audio (np.ndarray): Input audio signal.
        n (int): Total samples in audio.
        L (int): Total samples in frame grid.
        frame_size (int): Number of samples per frame.
        num_frames (int): Total frames to extract.

    Returns:
        np.ndarray: Array of shape (num_frames, frame_size).
    """
    n = len(audio) # total samples in audio
    L = frame_size * num_frames  # total samples in frame grid
    k = np.arange(L)
    indices = (k * n) // L
    stretched = audio[indices]
    frames = stretched.reshape(num_frames, frame_size)
    processing_log.append(f"Stretched/compressed audio to fit {num_frames} frames of size {frame_size}")
    return frames

def slice_slide(audio, frame_size, num_frames, overlap):
    """
    Slice audio into overlapping frames.

    Args:
        audio (np.ndarray): Input audio signal.
        frame_size (int): Number of samples per frame.
        num_frames (int): Total frames to extract.
        overlap (float): Overlap between frames (0.0 - 1.0).

    Returns:
        np.ndarray: Array of shape (num_frames, frame_size).
    """
    hop_size = int(frame_size * (1 - overlap))
    frames = []
    start = 0
    for _ in range(num_frames):
        end = start + frame_size
        if end > len(audio):
            # Zero-pad if audio is too short
            frame = np.zeros(frame_size)
            available = len(audio) - start
            if available > 0:
                frame[:available] = audio[start:start+available]
        else:
            frame = audio[start:end]
        frames.append(frame)
        start += hop_size
    processing_log.append(f"Sliced audio into {num_frames} overlapping frames of size {frame_size} with {overlap*100}% overlap")
    return np.array(frames)

def slice_cycle(audio, sr, frame_size, num_frames):
    """
    Slice audio into frames based on detected pitch cycles.

    Args:
        audio (np.ndarray): Input audio signal.
        sr (int): Sample rate of the audio signal.
        frame_size (int): Number of samples per frame.
        num_frames (int): Total frames to extract.

    Returns:
        np.ndarray: Array of shape (num_frames, frame_size).
    """
    f0_t = track_f0(audio, sr, hop_length=256)
    start = 0
    frames = []
    hop = 2
    for f0 in f0_t:
        if np.isnan(f0):
            continue  # Ignore frames where f0 is not detected (NaN)

        samples_per_cycle = int(sr / f0)

         # Snap start to nearest zero crossing
        start = nearest_zero_crossing(audio, start)

        end = start + (samples_per_cycle * hop) # extract 4 cycles to ensure enough data
        if end > len(audio):
            break

        end = nearest_zero_crossing(audio, end)

        cycle = audio[start:end]
        if len(cycle) < 2:
            continue

        # Resample to fixed frame size
        cycle_resampled = np.interp(
            np.linspace(0, 1, frame_size),
            np.linspace(0, 1, len(cycle)),
            cycle
        )
        frames.append(cycle_resampled)

        start = end -  (samples_per_cycle * (hop - 1))  # move to next cycle

        if len(frames) >= num_frames:
            break
    processing_log.append(f"Sliced audio into {len(frames)} pitch-synced frames of size {frame_size}")
    return np.array(frames)


def spectral_to_wavetable(frames, fft_size=None, return_spectra=False, smoothing_factor=0.0, output_frame_size=2048):
    """
    Convert frames into a wavetable: spectral resynthesis with phase alignment
    and reduction to single-cycle waveforms, downsampled to output_frame_size.

    Args:
        frames (np.ndarray): Array of shape (num_frames, frame_size).
        fft_size (int): FFT size for spectral analysis (defaults to frame_size).
        return_spectra (bool): If True, also return the spectral data for plotting.
        smoothing_factor (float): Smoothing factor for magnitudes between frames (0.0 - 1.0).
        output_frame_size (int): Size of each output waveform in the wavetable (default 2048).

    Returns:
        np.ndarray: Wavetable array of shape (num_frames, output_frame_size) where each
                    row is a normalized single-cycle waveform.
        dict (optional): Spectral data if return_spectra=True, containing:
                        - 'magnitudes': original magnitude spectra
                        - 'smoothed_magnitudes': smoothed magnitude spectra
                        - 'phases': phase spectra
                        - 'aligned_phases': phase-aligned spectra
                        - 'freqs': frequency bins
    """
    num_frames, frame_size = frames.shape
    fft_size = fft_size or frame_size # default to frame_size CHECK EFFECTS OF DEFAULTING TO frame_size <-------------------------------------------------------------------------
    if fft_size < frame_size:
        raise ValueError("FFT size must be >= frame_size")

    if not (0.0 <= smoothing_factor <= 1.0):
        raise ValueError("smoothing_factor must be between 0.0 and 1.0")

    # Hanning window
    window = np.hanning(frame_size) 

    wavetable = []

    # Phase accumulator
    prev_phase = np.zeros(fft_size // 2 + 1)

    # For smoothing magnitudes
    prev_smoothed_magnitude = None

    # Storage for spectral data if requested
    if return_spectra:
        magnitudes = []
        smoothed_magnitudes = []
        phases = []
        aligned_phases = []
        freqs = np.fft.rfftfreq(fft_size, d=1)  # Normalized frequencies

    for i in range(num_frames):
        frame = frames[i] * window

        # FFT
        spectrum = np.fft.rfft(frame, n=fft_size)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Apply smoothing to magnitude
        if i == 0:
            smoothed_magnitude = magnitude
        else:
            smoothed_magnitude = smoothing_factor * prev_smoothed_magnitude + (1 - smoothing_factor) * magnitude
        prev_smoothed_magnitude = smoothed_magnitude

        # Phase alignment
        phase_diff = phase - prev_phase
        phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
        aligned_phase = prev_phase + phase_diff
        prev_phase = aligned_phase

        # Store spectral data if requested
        if return_spectra:
            magnitudes.append(magnitude)
            smoothed_magnitudes.append(smoothed_magnitude)
            phases.append(phase)
            aligned_phases.append(aligned_phase)

        # Reconstruct spectrum using smoothed magnitude
        new_spectrum = smoothed_magnitude * np.exp(1j * phase)
        resynth = np.fft.irfft(new_spectrum, n=fft_size)

        # Downsample to output_frame_size
        if fft_size != output_frame_size:
            x_old = np.linspace(0, 1, fft_size)
            x_new = np.linspace(0, 1, output_frame_size)
            cycle = np.interp(x_new, x_old, resynth[:fft_size])
        else:
            cycle = resynth[:fft_size]

        # Normalize to [-1, 1]
        cycle /= np.max(np.abs(cycle) + 1e-9)

        wavetable.append(cycle)

    if return_spectra:
        spectral_data = {
            'magnitudes': np.array(magnitudes),
            'smoothed_magnitudes': np.array(smoothed_magnitudes),
            'phases': np.array(phases),
            'aligned_phases': np.array(aligned_phases),
            'freqs': freqs
        }
        return np.array(wavetable), spectral_data
    processing_log.append(f"Spectral resynthesis complete: {num_frames} frames, output size {output_frame_size}")
    return np.array(wavetable)