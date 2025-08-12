from typing import List
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


def play_note(note: Note, actx: AudioContext, num_in_parallel: int = 1):
    instrument: Instrument = note.instrument

    waveforms: dict[str, np.ufunc] = {
        "sine": np.sin,
        "square": np.frompyfunc(lambda x: np.sign(np.sin(x)), 1, 1),
        "triangle": np.frompyfunc(lambda x: 2 / np.pi * np.arcsin(np.sin(x)), 1, 1),
        "sawtooth": np.frompyfunc(
            lambda x: 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5)), 1, 1
        ),
        "reverse_sawtooth": np.frompyfunc(
            lambda x: -2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5)), 1, 1
        ),
    }

    waveform: str = instrument.waveform
    adsr: List[float] = instrument.adsr

    wavefunc: np.ufunc = waveforms[waveform]

    attack_samples = int(adsr[0] * actx.sample_rate)
    decay_samples = int(adsr[1] * actx.sample_rate)
    release_samples = int(adsr[3] * actx.sample_rate)

    duration = note.duration * (60 / note.tempo)

    note_abs_length = int(duration * actx.sample_rate) + release_samples

    sustain_samples = max(
        note_abs_length - (attack_samples + decay_samples + release_samples), 0
    )

    attack_env = np.linspace(0, 1, attack_samples, endpoint=False)
    decay_env = np.linspace(1, adsr[2], decay_samples, endpoint=False)
    sustain_env = np.full(sustain_samples, adsr[2])
    release_env = np.linspace(adsr[2], 0, release_samples, endpoint=True)

    envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])

    if len(envelope) < note_abs_length:
        envelope = np.pad(envelope, (0, note_abs_length - len(envelope)))
    else:
        envelope = envelope[:note_abs_length]

    t = np.arange(note_abs_length) / actx.sample_rate
    freq = note_to_freq(note.note)

    samples: np.ndarray = wavefunc(2 * np.pi * freq * t).astype(np.float32)
    samples *= envelope * (1 / num_in_parallel)

    mixdown_result_length = max(len(actx.mixdown), actx.mixdown_ptr + note_abs_length)
    mixdown_result = np.zeros(mixdown_result_length, dtype=np.float32)

    mixdown_result[: len(actx.mixdown)] += actx.mixdown

    mixdown_result[actx.mixdown_ptr : actx.mixdown_ptr + note_abs_length] += samples

    actx.mixdown = mixdown_result
    actx.mixdown_ptr += note_abs_length - release_samples
