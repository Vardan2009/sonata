import traceback
import sys
import os

import colorama
from colorama import Fore

import lexer
import parser
import interpreter
import synthesis
import structures

import error


def execute_code(src: str, filename: str) -> None:
    try:
        tokens = lexer.tokenize_source(filename, src)
        _parser = parser.Parser(filename, tokens)

        root: parser.ASTNode = _parser.parse()

        ctx = interpreter.InterpreterContext(filename, 60)
        actx = synthesis.AudioContext()

        audio_tree: structures.SequenceValue = interpreter.visit_assert_type(
            root, structures.SequenceValue, ctx
        )

        synthesis.play_result(audio_tree, actx)
    except error.SonataError as e:
        e.print(src)


def repl() -> int:
    while True:
        print("Sonata % ", end="")
        line = input().strip()

        if line == "exit":
            return 0

        execute_code(line, "<stdin>")


def execute_file(path: str) -> int:
    with open(path, "r") as f:
        content = f.read()
        execute_code(content, os.path.basename(path))
        return 0


def main() -> int:
    colorama.init()

    if len(sys.argv) == 1:
        return repl()
    elif len(sys.argv) == 2:
        return execute_file(sys.argv[1])
    else:
        print(
            f"Sonata {Fore.GREEN}USAGE{Fore.RESET}: {Fore.BLUE}sonata{Fore.RESET} [filepath]"
        )
        return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        error.SonataError(error.SonataErrorType.INTERNAL_ERROR, str(e)).print("")
        traceback.print_exc()
