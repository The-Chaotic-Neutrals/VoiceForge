"""
Spectrogram utilities for VR vocal separation.
From Mangio-RVC-Fork/lib/uvr5_pack/lib_v5/spec_utils.py
"""
import numpy as np
import librosa
import threading


def crop_center(h1, h2):
    """Crop h1 to match h2's time dimension."""
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]
    return h1


def wave_to_spectrogram(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):
    """Convert waveform to spectrogram."""
    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = librosa.stft(wave_left, n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])
    return spec


def wave_to_spectrogram_mt(wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False):
    """Multi-threaded wave to spectrogram conversion."""
    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left = None
    
    def run_thread(**kwargs):
        nonlocal spec_left
        spec_left = librosa.stft(**kwargs)

    thread = threading.Thread(
        target=run_thread,
        kwargs={"y": wave_left, "n_fft": n_fft, "hop_length": hop_length},
    )
    thread.start()
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)
    thread.join()

    spec = np.asfortranarray([spec_left, spec_right])
    return spec


def combine_spectrograms(specs, mp):
    """Combine spectrograms from multiple bands."""
    import math
    
    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][
            :, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l
        ]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too many bins")

    # lowpass filter
    if mp.param["pre_filter_start"] > 0:
        if bands_n == 1:
            spec_c = fft_lp_filter(
                spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"]
            )
        else:
            gp = 1
            for b in range(
                mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]
            ):
                g = math.pow(
                    10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                )
                gp = g
                spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


def fft_lp_filter(spec, bin_start, bin_stop):
    """Low-pass filter."""
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0
    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    """High-pass filter."""
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0
    return spec


def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse):
    """Convert spectrogram to waveform."""
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray(
            [np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)]
        )
    elif mid_side_b2:
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
    """Combine spectrogram bands and convert to wave."""
    bands_n = len(mp.param["band"])
    offset = 0
    wave = None

    for d in range(1, bands_n + 1):
        print(f"[UVR5] Processing band {d}/{bands_n}...", flush=True)
        bp = mp.param["band"][d]
        spec_s = np.ndarray(
            shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex
        )
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
            :, offset : offset + h, :
        ]
        offset += h

        if d == bands_n:  # highest band
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                    :, :extra_bins_h, :
                ]
            if bp.get("hpf_start", 0) > 0:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp.get("hpf_stop", bp["hpf_start"]) - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    mp.param["mid_side"],
                    mp.param["mid_side_b2"],
                    mp.param["reverse"],
                )
            else:
                wave = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lowest band
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                    orig_sr=bp["sr"],
                    target_sr=sr,
                    res_type="sinc_fastest",
                )
            else:  # mid bands
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp.get("hpf_stop", bp["hpf_start"]) - 1)
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
                wave = librosa.resample(
                    wave2, orig_sr=bp["sr"], target_sr=sr, res_type="scipy"
                )

    return wave.T


def mirroring(a, spec_m, input_high_end, mp):
    """Mirror missing frequency range."""
    if "mirroring" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))
        return np.where(
            np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror
        )
    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)
        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)
    
    return input_high_end
