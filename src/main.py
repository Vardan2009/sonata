import traceback
import os
import argparse
import hashlib
import time

from typing import Union, Callable, Optional

import colorama
from colorama import Fore

from pyaudio import PyAudio

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, DirModifiedEvent

import lexer
import parser
import interpreter
import synthesis
import structures

import error

import config

SONATA_VERSION: str = "0.3"


def execute_code(src: str, filename: str) -> Optional[structures.SequenceValue]:
    try:
        tokens = lexer.tokenize_source(filename, src)
        _parser = parser.Parser(filename, tokens)

        root: parser.ASTNode = _parser.parse()

        ctx = interpreter.InterpreterContext(filename, 60)

        audio_tree: structures.SequenceValue = interpreter.visit_assert_type(
            root, structures.SequenceValue, ctx
        )

        return audio_tree
    except error.SonataError as e:
        e.print(src)
        return None


def repl() -> int:
    actx: synthesis.AudioContext = synthesis.AudioContext()

    while True:
        print(f"{Fore.YELLOW}Sonata %{Fore.RESET} ", end="")
        line = input().strip()

        if line == "exit":
            return 0
        elif len(line) == 0:
            continue

        audio_tree = execute_code(line, "<stdin>")
        if audio_tree:
            actx.clear()
            synthesis.play_result(audio_tree, actx)


def execute_file(path: str) -> Optional[structures.SequenceValue]:
    try:
        with open(path, "r") as f:
            content = f.read()
            audio_tree = execute_code(content, os.path.basename(path))
            return audio_tree
    except FileNotFoundError:
        print(f"Sonata: {Fore.RED}File not found{Fore.RESET}")
        return None


class LoopHandler(FileSystemEventHandler):
    def __init__(self, filename: str, callback: Callable[[], None]) -> None:
        self.filename: str = filename
        self.callback: Callable[[], None] = callback
        self.last_hash: str = ""
        super().__init__()

    def on_modified(self, event: Union[FileModifiedEvent, DirModifiedEvent]):
        if (
            event.is_directory
            or event.event_type != "modified"
            or event.src_path != self.filename
        ):
            return

        h = hashlib.md5()
        with open(self.filename, "rb") as f:
            h.update(f.read())
        hash: str = h.hexdigest()

        if hash != self.last_hash:
            print(f"{Fore.GREEN}File change detected!{Fore.RESET}")
            self.callback()
            self.last_hash = hash


def loop_file(path: str) -> int:
    directory, _ = os.path.split(path)

    actx: synthesis.AudioContext = synthesis.AudioContext()

    def change_handler():
        audio_tree = execute_file(path)

        if audio_tree:
            actx.clear()
            try:
                audio_tree.mixdown(actx, 1)
                print("File changes applied!")
            except Exception as e:
                error.SonataError(error.SonataErrorType.INTERNAL_ERROR, str(e)).print(
                    ""
                )
                traceback.print_exc()

    observer = Observer()
    observer.schedule(
        LoopHandler(path, change_handler), path=directory, recursive=False
    )
    observer.start()

    print(f"Started listening to file changes to {path}...")

    change_handler()  # initial start

    p = PyAudio()
    stream = p.open(
        format=config.SAMPLE_TYPE,
        channels=config.CHANNELS,
        rate=config.SAMPLE_RATE,
        output=True,
    )

    try:
        while True:
            if len(actx.mixdown) > 0 and not config.NO_PLAY:
                data = actx.mixdown.tobytes()
                chunk_size: int = 1024
                for i in range(0, len(data), chunk_size):
                    stream.write(data[i : i + chunk_size])
            else:
                time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()

    stream.stop_stream()
    stream.close()
    p.terminate()

    observer.join()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Block-based domain-specific language for structured music composition"
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Sonata version {SONATA_VERSION}",
        help="show sonata version number and exit",
    )

    parser.add_argument(
        "file", nargs="?", help="file to open (leave empty to start REPL)"
    )

    parser.add_argument("-l", "--loop", action="store_true", help="enable loop mode")
    parser.add_argument(
        "-np", "--no-play", action="store_true", help="don't play result"
    )

    args = parser.parse_args()
    colorama.init()

    config.NO_PLAY = args.no_play

    if args.file:
        if args.loop:
            return loop_file(args.file)

        actx: synthesis.AudioContext = synthesis.AudioContext()
        audio_tree: Optional[synthesis.SequenceValue] = execute_file(args.file)

        if not audio_tree:
            return 1

        synthesis.play_result(audio_tree, actx)
        return 0
    else:
        return repl()


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        error.SonataError(error.SonataErrorType.INTERNAL_ERROR, str(e)).print("")
        traceback.print_exc()
