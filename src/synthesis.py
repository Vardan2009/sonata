from structures import Note, Instrument, AudioContext, SequenceValue

import numpy as np

from pyaudio import PyAudio, paFloat32


def note_to_freq(note: str) -> float:
    note = note.upper()

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
    return freq


def play_result(root: SequenceValue, actx: AudioContext):
    root.mixdown(actx, 1)

    if len(actx.mixdown) != 0:
        p = PyAudio()

        stream = p.open(
            format=paFloat32, channels=1, rate=actx.sample_rate, output=True
        )

        stream.write(actx.mixdown.tobytes())

        stream.stop_stream()
        stream.close()

        p.terminate()


def _square_wave(x: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(x))


def _triangle_wave(x: np.ndarray) -> np.ndarray:
    return 2 / np.pi * np.arcsin(np.sin(x))


def _sawtooth_wave(x: np.ndarray) -> np.ndarray:
    return 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))


def _reverse_sawtooth_wave(x: np.ndarray) -> np.ndarray:
    return -2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))


WAVEFORMS: dict[str, np.ufunc] = {
    "sine": np.sin,
    "square": np.frompyfunc(_square_wave, 1, 1),
    "triangle": np.frompyfunc(_triangle_wave, 1, 1),
    "sawtooth": np.frompyfunc(_sawtooth_wave, 1, 1),
    "reverse_sawtooth": np.frompyfunc(_reverse_sawtooth_wave, 1, 1),
}


def play_note(note: "Note", actx: "AudioContext", num_in_parallel: int = 1):
    instrument: Instrument = note.instrument
    adsr = instrument.adsr
    wavefunc = WAVEFORMS[instrument.waveform]

    attack_samples = int(adsr[0] * actx.sample_rate)
    decay_samples = int(adsr[1] * actx.sample_rate)
    release_samples = int(adsr[3] * actx.sample_rate)

    duration = note.duration * (60 / note.tempo)
    note_abs_length = int(duration * actx.sample_rate) + release_samples

    sustain_samples = max(
        note_abs_length - (attack_samples + decay_samples + release_samples), 0
    )

    envelope = np.empty(note_abs_length, dtype=np.float32)
    
    pos = 0
    envelope[pos : pos + attack_samples] = np.linspace(
        0, 1, attack_samples, endpoint=False, dtype=np.float32
    )

    pos += attack_samples
    envelope[pos : pos + decay_samples] = np.linspace(
        1, adsr[2], decay_samples, endpoint=False, dtype=np.float32
    )
    
    pos += decay_samples
    envelope[pos : pos + sustain_samples] = adsr[2]

    pos += sustain_samples
    envelope[pos : pos + release_samples] = np.linspace(
        adsr[2], 0, release_samples, endpoint=True, dtype=np.float32
    )

    t = np.arange(note_abs_length, dtype=np.float32) * (
        2 * np.pi * note_to_freq(note.note) / actx.sample_rate
    )

    samples = wavefunc(t).astype(np.float32)
    samples *= envelope
    samples *= 1.0 / num_in_parallel

    needed_len = actx.mixdown_ptr + note_abs_length
    if needed_len > len(actx.mixdown):
        actx.mixdown = np.pad(
            actx.mixdown, (0, needed_len - len(actx.mixdown)), constant_values=0
        )

    actx.mixdown[actx.mixdown_ptr : actx.mixdown_ptr + note_abs_length] += samples
    actx.mixdown_ptr += note_abs_length - release_samples
