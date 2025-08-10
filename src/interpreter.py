from typing import Dict, Union, Optional, List, cast, TypeAlias

import lexer
import parser

from error import SonataError, SonataErrorType


Instrument: TypeAlias = Dict[str, List[parser.ASTNode]]
Value: TypeAlias = Optional[Union[int, float, parser.ASTNode, Instrument]]


class Scope:
    tempo: Optional[float]
    instrument: Optional[Instrument]
    symbols: Dict[str, Value]

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
        return 0

    def __init__(self, file: str, initial_tempo: float):
        self.file = file
        self.scope_stack = [Scope(initial_tempo, {})]


def visit_node(node: parser.ASTNode, ctx: InterpreterContext) -> Value:
    match type(node):
        case parser.SequenceNode:
            sequence_node: parser.SequenceNode = cast(parser.SequenceNode, node)

            if sequence_node.is_parallel:
                print("== PARALLEL ==")
            for n in sequence_node.contents:
                visit_node(n, ctx)
            if sequence_node.is_parallel:
                print("==============")

        case parser.NumberNode:
            return cast(parser.NumberNode, node).value

        case parser.SymbolNode:
            return ctx.get_symbol(
                cast(parser.SymbolNode, node).symbol, node.line, node.column
            )

        case parser.NoteNode:
            note_node = cast(parser.NoteNode, node)
            duration: Value = visit_node(note_node.duration, ctx)
            print(
                f"*Played note {note_node.note_symbol}:{duration} beats with tempo {ctx.get_tempo()} BPM*"
            )
            return None

        case parser.BinOpNode:
            binop_node: parser.BinOpNode = cast(parser.BinOpNode, node)

            if binop_node.op == lexer.TokenType.EQ:
                if type(binop_node.left) is not parser.SymbolNode:
                    raise SonataError(
                        SonataErrorType.SYNTAX_ERROR,
                        "LHS of assignment should be symbol",
                        ctx.file,
                        binop_node.left.line,
                        binop_node.left.column,
                    )

                ctx.set_symbol(
                    binop_node.left.symbol,
                    visit_node(binop_node.right, ctx),
                    binop_node.line,
                    binop_node.column,
                )
                return None

            left_val: Value = visit_node(binop_node.left, ctx)
            right_val: Value = visit_node(binop_node.right, ctx)

            if (type(left_val) is not int and type(left_val) is not float) or (
                type(right_val) is not int and type(right_val) is not float
            ):
                raise SonataError(
                    SonataErrorType.SYNTAX_ERROR,
                    "Invalid binary operation parameters",
                    ctx.file,
                    binop_node.left.line,
                    binop_node.left.column,
                )

            match binop_node.op:
                case lexer.TokenType.PLUS:
                    return left_val + right_val
                case lexer.TokenType.MINUS:
                    return left_val - right_val
                case lexer.TokenType.STAR:
                    return left_val * right_val
                case lexer.TokenType.SLASH:
                    return left_val / right_val
                case _:
                    raise SonataError(
                        SonataErrorType.SYNTAX_ERROR,
                        "Invalid binary operator",
                        ctx.file,
                        binop_node.left.line,
                        binop_node.left.column,
                    )

        case parser.TempoNode:
            tempo: Value = visit_node(cast(parser.TempoNode, node).tempo, ctx)

            if type(tempo) is not float and type(tempo) is not int:
                raise SonataError(
                    SonataErrorType.SYNTAX_ERROR,
                    "Tempo should be a number",
                    ctx.file,
                    node.line,
                    node.column,
                )

            ctx.set_tempo(tempo)
            return tempo

        case parser.DefineNode:
            define_node = cast(parser.DefineNode, node)
            ctx.set_symbol(
                define_node.alias,
                define_node.value,
                define_node.line,
                define_node.column,
            )
            return None

        case parser.UseNode:
            instrument = cast(Instrument, visit_node(node, ctx))
            ctx.set_instrument(instrument)
            return None
