from typing import Any, cast

import lexer
import parser

from structures import Instrument, Value, InterpreterContext, SequenceValue, Note
from error import SonataError, SonataErrorType


def visit_assert_type(node: parser.ASTNode, t: type, ctx: InterpreterContext) -> Any:
    value: Value = visit_node(node, ctx)

    if type(value) is not t:
        raise SonataError(
            SonataErrorType.SYNTAX_ERROR,
            f"Expected a value of type {t.__name__}",
            ctx.file,
            node.line,
            node.column,
        )

    return value


def visit_node(node: parser.ASTNode, ctx: InterpreterContext) -> Value:
    match type(node):
        case parser.SequenceNode:
            sequence_node: parser.SequenceNode = cast(parser.SequenceNode, node)

            sequence: SequenceValue = SequenceValue([], sequence_node.is_parallel)

            for n in sequence_node.contents:
                val: Value = visit_node(n, ctx)

                if type(val) is Note or type(val) is SequenceValue:
                    sequence.notes.append(val)

            return sequence

        case parser.NumberNode:
            return cast(parser.NumberNode, node).value

        case parser.StringNode:
            return cast(parser.StringNode, node).value

        case parser.SymbolNode:
            sym = ctx.get_symbol(
                cast(parser.SymbolNode, node).symbol, node.line, node.column
            )

            if isinstance(sym, parser.ASTNode):
                return visit_node(sym, ctx)

            return sym

        case parser.NoteNode:
            note_node = cast(parser.NoteNode, node)
            duration: float = visit_assert_type(note_node.duration, float, ctx)

            return Note(note_node.note_symbol, duration, ctx)

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

        case parser.UseNode:
            instrument: Instrument = ctx.get_instrument_conf(
                cast(parser.UseNode, node).config, node.line, node.column
            )
            ctx.set_instrument(instrument)
            return None

        case parser.InstrumentNode:
            instrument_node = cast(parser.InstrumentNode, node)
            result: Instrument = Instrument(instrument_node, ctx)
            ctx.set_instrument_conf(
                instrument_node.instrument_name,
                result,
                instrument_node.line,
                instrument_node.column,
            )

        case parser.RepeatNode:
            sequence: SequenceValue = SequenceValue([], False)

            repeat_node = cast(parser.RepeatNode, node)

            times: int = int(visit_assert_type(repeat_node.times, float, ctx))

            for _ in range(times):
                val: Value = visit_node(repeat_node.root, ctx)

                if type(val) is Note or type(val) is SequenceValue:
                    sequence.notes.append(val)

            return sequence

        case parser.DefineNode:
            define_node: parser.DefineNode = cast(parser.DefineNode, node)

            ctx.set_symbol(
                define_node.alias,
                define_node.root,
                define_node.line,
                define_node.column,
            )

        case _:
            raise SonataError(
                SonataErrorType.INTERNAL_ERROR,
                "Unhandled node type",
                ctx.file,
                node.line,
                node.column,
            )
