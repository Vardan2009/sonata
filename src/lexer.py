from enum import Enum
from typing import List, Union, Dict

from error import SonataError, SonataErrorType

TokenValue = Union[str, int, float]


class TokenType(Enum):
    LBRACE = 0
    RBRACE = 1
    LPAREN = 2
    RPAREN = 3
    KEYWORD = 4
    IDENTIFIER = 5
    INSTRUMENT_CONFIG = 6
    NUMBER = 7
    COMMA = 8
    LSQR = 9
    RSQR = 10

    PLUS = 11
    MINUS = 12
    STAR = 13
    SLASH = 14
    PERCENT = 15
    EQ = 16

    COLON = 17

    STRING = 18


class Token:
    type: TokenType
    value: TokenValue
    line: int
    column: int

    def __init__(self, type: TokenType, value: TokenValue, line: int, column: int):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"{self.type} -> ( {self.value} )"


def tokenize_source(filename: str, src: str) -> List[Token]:
    result: List[Token] = []
    ptr = 0
    line = 1
    column = 0
    length = len(src)

    single_char_tokens: Dict[str, TokenType] = {
        "{": TokenType.LBRACE,
        "}": TokenType.RBRACE,
        "(": TokenType.LPAREN,
        ")": TokenType.RPAREN,
        ",": TokenType.COMMA,
        "[": TokenType.LSQR,
        "]": TokenType.RSQR,
        "+": TokenType.PLUS,
        "-": TokenType.MINUS,
        "*": TokenType.STAR,
        "/": TokenType.SLASH,
        "=": TokenType.EQ,
        ":": TokenType.COLON,
    }

    all_keywords: List[str] = ["tempo", "define", "use", "repeat", "instrument"]
    instrument_configs_keywords: List[str] = ["adsr", "waveform", "sample"]

    def tokenize_number() -> None:
        nonlocal line
        nonlocal column

        value_str: str = ""
        is_float: bool = False

        while ptr < length and (
            src[ptr].isdigit() or (not is_float and src[ptr] == ".")
        ):
            if src[ptr] == ".":
                is_float = True
            value_str += src[ptr]
            next_char()

        result.append(
            Token(
                TokenType.NUMBER,
                float(value_str) if is_float else int(value_str),
                line,
                column,
            )
        )

    def tokenize_string() -> None:
        nonlocal line
        nonlocal column

        value_str: str = ""

        next_char()

        while ptr < length and src[ptr] != '"':
            value_str += src[ptr]
            next_char()

        next_char()

        result.append(Token(TokenType.STRING, value_str, line, column))

    def tokenize_identifier() -> None:
        nonlocal line
        nonlocal column

        value_str: str = ""

        while ptr < length and (
            src[ptr].isalnum() or src[ptr] == "_" or src[ptr] == "#"
        ):
            value_str += src[ptr]
            next_char()

        token_type: TokenType = TokenType.IDENTIFIER
        if value_str in all_keywords:
            token_type = TokenType.KEYWORD

        elif value_str in instrument_configs_keywords:
            token_type = TokenType.INSTRUMENT_CONFIG

        result.append(Token(token_type, value_str, line, column))

    def next_char() -> None:
        nonlocal ptr
        nonlocal column

        ptr += 1
        column += 1

    def skip_whitespace() -> None:
        nonlocal ptr
        nonlocal line
        nonlocal column

        while ptr < length and src[ptr].isspace():
            if src[ptr] == "\n":
                line += 1
                column = 0
            next_char()

    while ptr < length:
        char = src[ptr]
        if char in single_char_tokens:
            result.append(Token(single_char_tokens[char], char, line, column))
            next_char()
        elif char == '"':
            tokenize_string()
        elif char.isdigit():
            tokenize_number()
        elif char.isalpha() or char == "_":
            tokenize_identifier()
        elif char.isspace():
            skip_whitespace()
        else:
            raise SonataError(
                SonataErrorType.SYNTAX_ERROR,
                f"Unknown character `{char}`",
                filename,
                line,
                column,
            )

    return result
