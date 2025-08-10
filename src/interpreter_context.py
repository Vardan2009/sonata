from typing import Optional, Dict, List, TypeAlias, Union
from parser import ASTNode

from error import SonataError, SonataErrorType

Instrument: TypeAlias = Dict[str, List[ASTNode]]
InstrumentCompiled: TypeAlias = Dict[str, List[float]]
Value: TypeAlias = Optional[Union[int, float, str, ASTNode, InstrumentCompiled]]

class Scope:
    tempo: Optional[float]
    instrument: Optional[InstrumentCompiled] = None
    symbols: Dict[str, Value]
    defined_instruments: Dict[str, InstrumentCompiled] = {}

    def __init__(self, tempo: Optional[float] = None, symbols: Dict[str, Value] = {}):
        self.tempo = tempo
        self.symbols = symbols


class InterpreterContext:
    file: str
    scope_stack: List[Scope]

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

    def set_symbol(
        self, symbol_name: str, value: Value, line: int, column: int
    ) -> None:
        for scope in reversed(self.scope_stack):
            if symbol_name in scope.symbols:
                scope.symbols[symbol_name] = value
        self.scope_stack[-1].symbols[symbol_name] = value


    def get_instrument_conf(self, symbol_name: str, line: int, column: int) -> InstrumentCompiled:
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
        self, symbol_name: str, value: InstrumentCompiled, line: int, column: int
    ) -> None:
        self.scope_stack[-1].defined_instruments[symbol_name] = value

    def set_tempo(self, new_tempo: float):
        self.scope_stack[-1].tempo = new_tempo

    def get_tempo(self) -> float:
        for scope in reversed(self.scope_stack):
            if scope.tempo:
                return scope.tempo
        return 0

    def set_instrument(self, new_instrument: InstrumentCompiled):
        self.scope_stack[-1].instrument = new_instrument

    def get_instrument(self) -> InstrumentCompiled:
        for scope in reversed(self.scope_stack):
            if scope.instrument:
                return scope.instrument
        return {"adsr": [0.1,0,1,0.1]}

    def __init__(self, file: str, initial_tempo: float):
        self.file = file
        self.scope_stack = [Scope(initial_tempo, {})]