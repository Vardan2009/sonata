from structures import Note, Instrument, AudioContext, SequenceValue
from typing_extensions import Literal
from typing import Tuple, Dict, Callable

import numpy as np

import config
from pyaudio import PyAudio

from scipy.signal import butter, filtfilt

filter_cache: Dict[
    Tuple[float, int, Literal["low", "high"]], Tuple[np.ndarray, np.ndarray]
] = {}


def pass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sample_rate: float,
    order: int,
    btype: Literal["low", "high"],
) -> np.ndarray:
    key = (cutoff_freq, order, btype)
    if key not in filter_cache:
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        filter_cache[key] = butter(order, normal_cutoff, btype, analog=False)
    b, a = filter_cache[key]
    return filtfilt(b, a, data)


note_freq_cache: Dict[str, float] = {}


def note_to_freq(note: str) -> float:
    if note[0] == "_":
        return 0

    if note in note_freq_cache:
        return note_freq_cache[note]

    note = note[0].upper() + note[1:]

    note_offsets = {
        "C": -9,
        "C#": -8,
        "Db": -8,
        "D": -7,
        "D#": -6,
        "Eb": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        "Gb": -3,
        "G": -2,
        "G#": -1,
        "Ab": -1,
        "A": 0,
        "A#": 1,
        "Bb": 1,
        "B": 2,
    }

    if len(note) == 2:
        name = note[0]
        octave = int(note[1])
    else:
        name = note[:2]
        octave = int(note[2])

    semitone_diff = note_offsets[name] + (octave - 4) * 12

    freq = 440 * (2 ** (semitone_diff / 12))
    note_freq_cache[note] = freq
    return freq


def play_result(root: SequenceValue, actx: AudioContext):
    root.mixdown(actx, 1)

    if len(actx.mixdown) != 0:
        p = PyAudio()

        stream = p.open(
            format=config.SAMPLE_TYPE,
            channels=config.CHANNELS,
            rate=config.SAMPLE_RATE,
            output=True,
        )

        data = actx.mixdown.tobytes()
        chunk_size: int = 1024
        for i in range(0, len(data), chunk_size):
            stream.write(data[i : i + chunk_size])

        stream.stop_stream()
        stream.close()

        p.terminate()


def square_wave(x: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(x)).astype(np.float32)


def triangle_wave(x: np.ndarray) -> np.ndarray:
    return (2 / np.pi * np.arcsin(np.sin(x))).astype(np.float32)


def sawtooth_wave(x: np.ndarray) -> np.ndarray:
    return (2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))).astype(np.float32)


def reverse_sawtooth_wave(x: np.ndarray) -> np.ndarray:
    return (-2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))).astype(np.float32)


WAVEFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sine": lambda x: np.sin(x).astype(np.float32),
    "square": square_wave,
    "triangle": triangle_wave,
    "sawtooth": sawtooth_wave,
    "reverse_sawtooth": reverse_sawtooth_wave,
}

envelopes_cache: Dict[
    Tuple[Tuple[float, float, float, float], float],
    Tuple[np.ndarray, int, int, int, int],
] = {}


def get_envelope(adsr: tuple[float, float, float, float], duration: float, sr: int):
    key = (adsr, duration)
    if key in envelopes_cache:
        return envelopes_cache[key]

    a_s = int(adsr[0] * sr)
    d_s = int(adsr[1] * sr)
    r_s = int(adsr[3] * sr)
    total_len = int(duration * sr) + r_s

    a_s = min(a_s, total_len - r_s)
    d_s = min(d_s, total_len - r_s - a_s)
    s_s = max(total_len - (a_s + d_s + r_s), 0)

    env = np.empty(total_len, dtype=np.float32)

    pos = 0
    if a_s:
        env[pos : pos + a_s] = np.arange(a_s, dtype=np.float32) / a_s
        pos += a_s
    if d_s:
        env[pos : pos + d_s] = 1 - (1 - adsr[2]) * (
            np.arange(d_s, dtype=np.float32) / d_s
        )
        pos += d_s
    if s_s:
        env[pos : pos + s_s] = adsr[2]
        pos += s_s
    if r_s:
        start_val = 1.0 if d_s == 0 else adsr[2]
        env[pos : pos + r_s] = start_val * (
            1 - np.arange(r_s, dtype=np.float32) / (r_s - 1 if r_s > 1 else 1)
        )

    result = (env, a_s, d_s, r_s, total_len)
    envelopes_cache[key] = result
    return result


t_cache: Dict[Tuple[float, float], np.ndarray] = {}


def get_t(length: float, freq: float, sr: float) -> np.ndarray:
    key = (length, freq)
    if key not in t_cache:
        t_cache[key] = np.arange(length, dtype=np.float32) * (2 * np.pi * freq / sr)
    return t_cache[key]


def play_note(note: "Note", actx: AudioContext, num_in_parallel: int = 1):
    instrument: Instrument = note.instrument
    adsr = instrument.adsr
    wavefunc = WAVEFORMS[instrument.waveform]

    duration = note.duration * (60 / note.tempo)

    (envelope, _, _, release_samples, note_abs_length) = get_envelope(
        adsr, duration, config.SAMPLE_RATE
    )

    t = get_t(note_abs_length, note_to_freq(note.note), config.SAMPLE_RATE)

    samples = wavefunc(t)
    samples *= envelope * note.volume * 0.7
    samples *= 1.0 / num_in_parallel

    if instrument.lowpass_freq and instrument.lowpass_order:
        samples = pass_filter(
            samples,
            instrument.lowpass_freq,
            config.SAMPLE_RATE,
            instrument.lowpass_order,
            "low",
        )

    if instrument.highpass_freq and instrument.highpass_order:
        samples = pass_filter(
            samples,
            instrument.highpass_freq,
            config.SAMPLE_RATE,
            instrument.highpass_order,
            "high",
        )

    left_gain = np.cos((note.pan + 1) * np.pi / 4)
    right_gain = np.sin((note.pan + 1) * np.pi / 4)
    stereo_samples = np.column_stack((samples * left_gain, samples * right_gain))

    needed_len = actx.mixdown_ptr + note_abs_length
    if needed_len > len(actx.mixdown):
        tmp = np.zeros((needed_len, 2), dtype=np.float32)
        tmp[: len(actx.mixdown), :] = actx.mixdown
        actx.mixdown = tmp

    actx.mixdown[actx.mixdown_ptr : actx.mixdown_ptr + note_abs_length, :] += (
        stereo_samples
    )
    actx.mixdown_ptr += note_abs_length - release_samples
