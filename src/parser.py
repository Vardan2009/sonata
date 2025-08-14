import lexer
import error

from typing import List, Dict, cast


class ASTNode:
    def __init__(self, line: int, column: int):
        self.line: int = line
        self.column: int = column

    def pretty_print(self, indent: int = 0) -> None:
        print(" " * (indent * 2), end="")


class SequenceNode(ASTNode):
    def __init__(
        self, is_parallel: bool, contents: List[ASTNode], line: int, column: int
    ):
        self.is_parallel: bool = is_parallel
        self.contents: List[ASTNode] = contents
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print("Parallel" if self.is_parallel else "Sequence")

        for node in self.contents:
            node.pretty_print(indent + 1)


class NumberNode(ASTNode):
    def __init__(self, value: float, line: int, column: int):
        self.value: float = value
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print(self.value)


class StringNode(ASTNode):
    def __init__(self, value: str, line: int, column: int):
        self.value: str = value
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print(f'"{self.value}"')


class SymbolNode(ASTNode):
    def __init__(self, symbol: str, line: int, column: int):
        self.symbol: str = symbol
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print(self.symbol)


class NoteNode(ASTNode):
    def __init__(self, note_symbol: str, duration: ASTNode, line: int, column: int):
        self.note_symbol: str = note_symbol
        self.duration: ASTNode = duration
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print(f"Note {self.note_symbol}")
        self.duration.pretty_print(indent + 1)


class BinOpNode(ASTNode):
    def __init__(
        self, op: lexer.TokenType, left: ASTNode, right: ASTNode, line: int, column: int
    ):
        self.left: ASTNode = left
        self.right: ASTNode = right
        self.op: lexer.TokenType = op
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print(f"BinOp {self.op}")
        self.left.pretty_print(indent + 1)
        self.right.pretty_print(indent + 1)


class TempoNode(ASTNode):
    def __init__(self, tempo: ASTNode, line: int, column: int):
        self.tempo: ASTNode = tempo
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print("Tempo")
        self.tempo.pretty_print(indent + 1)


class InstrumentNode(ASTNode):
    def __init__(
        self, name: str, config: Dict[str, List[ASTNode]], line: int, column: int
    ):
        self.instrument_name: str = name
        self.config: Dict[str, List[ASTNode]] = config
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print(f"Instrument {self.instrument_name}")

        for name, values in self.config.items():
            super().pretty_print(indent + 1)
            print(name)
            for value in values:
                value.pretty_print(indent + 2)


class UseNode(ASTNode):
    def __init__(self, config: str, line: int, column: int):
        self.config: str = config
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print(f"Use {self.config}")


class RepeatNode(ASTNode):
    def __init__(self, times: ASTNode, root: ASTNode, line: int, column: int):
        self.times: ASTNode = times
        self.root: ASTNode = root
        super().__init__(line, column)

    def pretty_print(self, indent: int = 0) -> None:
        super().pretty_print(indent)
        print("Repeat")
        self.times.pretty_print(indent + 1)
        self.root.pretty_print(indent + 1)


class DefineNode(ASTNode):
    def __init__(self, alias: str, root: ASTNode, line: int, column: int):
        self.alias: str = alias
        self.root: ASTNode = root
        super().__init__(line, column)


class VolumeNode(ASTNode):
    def __init__(self, volume: ASTNode, line: int, column: int):
        self.volume: ASTNode = volume
        super().__init__(line, column)


class Parser:
    precedence: Dict[lexer.TokenType, int] = {
        lexer.TokenType.PLUS: 1,
        lexer.TokenType.MINUS: 1,
        lexer.TokenType.STAR: 2,
        lexer.TokenType.SLASH: 2,
        lexer.TokenType.EQ: 0,
    }

    def __init__(self, file: str, tokens: List[lexer.Token]):
        self.tokens: List[lexer.Token] = tokens
        self.file: str = file
        self.ptr: int = 0

    def bound_check(self):
        if self.ptr >= len(self.tokens):
            raise error.SonataError(
                error.SonataErrorType.INTERNAL_ERROR, "Parser out of bounds"
            )

    def advance(self) -> lexer.Token:
        self.bound_check()
        self.ptr += 1
        return self.tokens[self.ptr - 1]

    def reverse(self) -> None:
        self.bound_check()
        self.ptr -= 1
        self.bound_check()

    def peek(self) -> lexer.Token:
        self.bound_check()
        return self.tokens[self.ptr]

    def expect(self, type: lexer.TokenType) -> lexer.Token:
        self.bound_check()
        cur_token: lexer.Token = self.peek()
        if cur_token.type != type:
            raise error.SonataError(
                error.SonataErrorType.SYNTAX_ERROR,
                f"Expected {type}, found {cur_token}",
                self.file,
                cur_token.line,
                cur_token.column,
            )
        return self.advance()

    def parse(self) -> ASTNode:
        if len(self.tokens) == 0:
            return SequenceNode(False, [], 0, 0)
        return self.parse_sequence()

    def parse_sequence(self) -> ASTNode:
        opening_token = self.peek()
        is_parallel = opening_token.type == lexer.TokenType.LSQR
        closing_token_type = (
            lexer.TokenType.RSQR if is_parallel else lexer.TokenType.RBRACE
        )

        if (
            opening_token.type != lexer.TokenType.LSQR
            and opening_token.type != lexer.TokenType.LBRACE
        ):
            raise error.SonataError(
                error.SonataErrorType.SYNTAX_ERROR,
                "Expected LSQR or LBRACE",
                self.file,
                opening_token.line,
                opening_token.column,
            )
        self.advance()

        contents: List[ASTNode] = []

        while self.peek().type != closing_token_type:
            contents.append(self.parse_statement())

        self.expect(closing_token_type)

        return SequenceNode(
            is_parallel, contents, opening_token.line, opening_token.column
        )

    def parse_statement(self) -> ASTNode:
        current_token = self.peek()

        match current_token.type:
            case lexer.TokenType.LBRACE | lexer.TokenType.LSQR:
                return self.parse_sequence()
            case lexer.TokenType.KEYWORD:
                return self.parse_command()
            case _:
                return self.parse_expression()

    def parse_command(self) -> ASTNode:
        command_token = self.peek()
        self.expect(lexer.TokenType.KEYWORD)

        match command_token.value:
            case "tempo":
                tempo_val: ASTNode = self.parse_expression()
                return TempoNode(tempo_val, command_token.line, command_token.column)
            case "use":
                config: str = cast(str, self.expect(lexer.TokenType.IDENTIFIER).value)
                return UseNode(config, command_token.line, command_token.column)
            case "repeat":
                times: ASTNode = self.parse_expression()
                root: ASTNode = self.parse_expression()
                return RepeatNode(times, root, command_token.line, command_token.column)
            case "instrument":
                return self.parse_instrument()
            case "define":
                alias: str = cast(str, self.expect(lexer.TokenType.IDENTIFIER).value)
                root: ASTNode = self.parse_expression()
                return DefineNode(alias, root, command_token.line, command_token.column)
            case "volume":
                value: ASTNode = self.parse_expression()
                return VolumeNode(value, command_token.line, command_token.column)
            case _:
                raise error.SonataError(
                    error.SonataErrorType.SYNTAX_ERROR,
                    "Invalid Command",
                    self.file,
                    command_token.line,
                    command_token.column,
                )

    def parse_primary(self) -> ASTNode:
        current_token = self.advance()

        match current_token.type:
            case lexer.TokenType.NUMBER:
                return NumberNode(
                    float(current_token.value),
                    current_token.line,
                    current_token.column,
                )
            case lexer.TokenType.STRING:
                return StringNode(
                    cast(str, current_token.value),
                    current_token.line,
                    current_token.column,
                )
            case lexer.TokenType.IDENTIFIER:
                if self.peek().type == lexer.TokenType.COLON:
                    self.advance()
                    duration: ASTNode = self.parse_expression()
                    return NoteNode(
                        cast(str, current_token.value),
                        duration,
                        current_token.line,
                        current_token.column,
                    )
                return SymbolNode(
                    cast(str, current_token.value),
                    current_token.line,
                    current_token.column,
                )
            case lexer.TokenType.LPAREN:
                return self.parse_expression()
            case lexer.TokenType.LSQR | lexer.TokenType.LBRACE:
                self.reverse()
                return self.parse_sequence()
            case _:
                raise error.SonataError(
                    error.SonataErrorType.SYNTAX_ERROR,
                    "Invalid Primary",
                    self.file,
                    current_token.line,
                    current_token.column,
                )

    def parse_expression(self, min_precedence: int = 0) -> ASTNode:
        left: ASTNode = self.parse_primary()

        while True:
            op: lexer.Token = self.peek()

            if op.type not in self.precedence:
                break

            precedence: int = self.precedence[op.type]

            if precedence < min_precedence:
                break

            self.advance()

            right = self.parse_expression(precedence + 1)

            left = BinOpNode(op.type, left, right, left.line, left.column)

        return left

    def parse_instrument(self) -> ASTNode:
        name: str = cast(str, self.expect(lexer.TokenType.IDENTIFIER).value)

        opening_token = self.peek()
        self.expect(lexer.TokenType.LPAREN)

        config: Dict[str, List[ASTNode]] = {}

        while self.peek().type != lexer.TokenType.RPAREN:
            config_name: str = cast(
                str, self.expect(lexer.TokenType.INSTRUMENT_CONFIG).value
            )
            values: List[ASTNode] = []

            current_token_type: lexer.TokenType = self.peek().type

            while (
                current_token_type != lexer.TokenType.RPAREN
                and current_token_type != lexer.TokenType.COMMA
            ):
                values.append(self.parse_expression())

                if self.peek().type == lexer.TokenType.COMMA:
                    self.advance()
                    break

                if self.peek().type == lexer.TokenType.RPAREN:
                    break

            config[config_name] = values

        self.expect(lexer.TokenType.RPAREN)

        return InstrumentNode(name, config, opening_token.line, opening_token.column)
