from typing import Optional, Dict, List, TypeAlias, Union, cast
from parser import InstrumentNode

from error import SonataError, SonataErrorType

import numpy as np


class Note:
    def __init__(self, note: str, duration: float, ictx: "InterpreterContext"):
        self.note: str = note
        self.duration: float = duration
        self.instrument: Instrument = ictx.get_instrument()

    def __str__(self) -> str:
        return f"{self.note}:{self.duration}"

    def mixdown(self, actx: "AudioContext") -> None:
        from synthesis import play_note

        play_note(self, actx)


class SequenceValue:
    def __init__(self, notes: List[Union[Note, "SequenceValue"]], parallel: bool):
        self.notes: List[Union[Note, "SequenceValue"]] = notes
        self.parallel = parallel

    def __str__(self) -> str:
        opening: str = "[" if self.parallel else "{"
        closing: str = "]" if self.parallel else "}"
        return f"{opening}{' '.join([str(x) for x in self.notes])}{closing}"

    def mixdown(self, actx: "AudioContext") -> None:
        for i, note in enumerate(self.notes):
            prev_mixdown_ptr = actx.mixdown_ptr

            note.mixdown(actx)

            if self.parallel and i != len(self.notes) - 1:
                actx.mixdown_ptr = prev_mixdown_ptr


Value: TypeAlias = Optional[Union[int, float, str, Note, SequenceValue]]


class Instrument:
    def __init__(
        self,
        node: Optional["InstrumentNode"] = None,
        ctx: Optional["InterpreterContext"] = None,
    ):
        self.adsr: List[float] = [0.1, 0.0, 1.0, 0.1]
        self.waveform: str = "sine"

        if not (node and ctx):
            return

        from interpreter import (
            visit_node,
        )  # kept here to avoid potential circular import

        for config_name, values in node.config.items():
            eval_values = [visit_node(v, ctx) for v in values]

            match config_name:
                case "waveform":
                    val = eval_values[0]
                    if not isinstance(val, str):
                        raise TypeError(
                            f"waveform must be a string, got {type(val).__name__}"
                        )
                    self.waveform = val
                case "adsr":
                    if not all(isinstance(x, (float, int)) for x in eval_values):
                        raise TypeError("adsr values must be numeric")
                    self.adsr = list(map(float, cast(List[float], eval_values)))
                case _:
                    raise SonataError(
                        SonataErrorType.NAME_ERROR,
                        f"No such instrument config entry `{config_name}`",
                        ctx.file,
                        node.line,
                        node.column,
                    )

    @classmethod
    def empty(cls) -> "Instrument":
        return cls()


class Scope:
    def __init__(self, tempo: Optional[float] = None, symbols: Dict[str, Value] = {}):
        self.tempo: Optional[float] = tempo
        self.instrument: Optional[Instrument] = None
        self.symbols: Dict[str, Value] = symbols
        self.defined_instruments: Dict[str, Instrument] = {}


class AudioContext:
    def __init__(self) -> None:
        self.sample_rate: int = 44100

        self.mixdown: np.ndarray = np.array([], dtype=np.float32)
        self.mixdown_ptr: int = 0


class InterpreterContext:
    def get_symbol(self, symbol_name: str, line: int, column: int) -> Value:
        for scope in reversed(self.scope_stack):
            if symbol_name in scope.symbols:
                return scope.symbols[symbol_name]
        raise SonataError(
            SonataErrorType.NAME_ERROR,
            f"Symbol {symbol_name} not defined",
            self.file,
            line,
            column,
        )

    def get_current_scope(self) -> Scope:
        return self.scope_stack[-1]

    def set_symbol(
        self, symbol_name: str, value: Value, line: int, column: int
    ) -> None:
        for scope in reversed(self.scope_stack):
            if symbol_name in scope.symbols:
                scope.symbols[symbol_name] = value
        self.scope_stack[-1].symbols[symbol_name] = value

    def get_instrument_conf(
        self, symbol_name: str, line: int, column: int
    ) -> Instrument:
        for scope in reversed(self.scope_stack):
            if symbol_name in scope.defined_instruments:
                return scope.defined_instruments[symbol_name]
        raise SonataError(
            SonataErrorType.NAME_ERROR,
            f"Instrument {symbol_name} not defined",
            self.file,
            line,
            column,
        )

    def set_instrument_conf(
        self, symbol_name: str, value: Instrument, line: int, column: int
    ) -> None:
        self.scope_stack[-1].defined_instruments[symbol_name] = value

    def set_tempo(self, new_tempo: float):
        self.scope_stack[-1].tempo = new_tempo

    def get_tempo(self) -> float:
        for scope in reversed(self.scope_stack):
            if scope.tempo:
                return scope.tempo
        return 0

    def set_instrument(self, new_instrument: Instrument):
        self.scope_stack[-1].instrument = new_instrument

    def get_instrument(self) -> Instrument:
        for scope in reversed(self.scope_stack):
            if scope.instrument:
                return scope.instrument

        self.scope_stack[-1].instrument = Instrument.empty()
        return self.scope_stack[-1].instrument

    def __init__(self, file: str, initial_tempo: float):
        self.file: str = file
        self.scope_stack: List[Scope] = [Scope(initial_tempo, {})]
