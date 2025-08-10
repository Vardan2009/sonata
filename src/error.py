from enum import Enum
import sys

from colorama import Fore, Style


class SonataErrorType(Enum):
    UNKNOWN_ERROR = 0
    INTERNAL_ERROR = 1
    SYNTAX_ERROR = 2
    NAME_ERROR = 3


class SonataError(Exception):
    line: int = 0
    column: int = 0
    file: str = "<no file>"
    type: SonataErrorType

    def __init__(
        self,
        type: SonataErrorType,
        message: str,
        file: str = "",
        line: int = -1,
        column: int = -1,
    ):
        self.file = file
        self.type = type
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self.message)

    def print(self, src: str) -> None:
        if self.line != -1:
            src_line = src[self.line - 1].lstrip()
            column_idx = self.column - (len(src[self.line - 1]) - len(src_line))
            caret_offset = len(str(self.line)) + 3 + column_idx

            print(
                f"Sonata {Fore.RED}{self.type.name}{Fore.RESET} at {Fore.BLUE}{self.file}:{self.line}{Fore.RESET}: {self.message}",
                file=sys.stderr,
            )
            print(
                f"\n{Style.DIM}{self.line} | {Style.NORMAL}{src.split('\n')[self.line - 1]}"
            )

            print(f"{' ' * caret_offset}{Fore.RED}^ HERE{Fore.RESET}\n")
        else:
            print(
                f"Sonata {Fore.RED}{self.type.name}{Fore.RESET}: {self.message}",
                file=sys.stderr,
            )
