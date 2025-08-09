import colorama

import lexer
import parser

import error


def repl() -> int:
    while True:

        filename: str = "test.sn"

        print("Sonata % ", end="")
        line = input().strip()

        if line == "exit":
            return 0

        try:
            tokens = lexer.tokenize_source(filename, line)
            _parser = parser.Parser(filename, tokens)

            root: parser.ASTNode = _parser.parse()

            root.pretty_print()
        except error.SonataError as e:
            e.print(line)


def main() -> int:
    colorama.init()
    return repl()


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        error.SonataError(
            error.SonataErrorType.INTERNAL_ERROR, str(e)).print("")
