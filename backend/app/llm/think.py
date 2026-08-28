"""Remove hidden reasoning tags from the local model's visible stream.

LM Studio exposes reasoning as literal ``<think>`` tags for this model, while
the client renders every response byte. A short carry buffer handles tags split
across streamed chunks.
"""

from __future__ import annotations

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


class ThinkStripper:
    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False

    def feed(self, piece: str) -> str:
        self._buffer += piece
        visible: list[str] = []

        while self._buffer:
            if self._inside:
                close_at = self._buffer.find(CLOSE_TAG)
                if close_at == -1:
                    # Keep a suffix in case "</think>" is split across chunks.
                    keep = min(len(CLOSE_TAG) - 1, len(self._buffer))
                    self._buffer = self._buffer[-keep:]
                    break
                self._buffer = self._buffer[close_at + len(CLOSE_TAG) :]
                self._inside = False
                continue

            open_at = self._buffer.find(OPEN_TAG)
            if open_at == -1:
                keep = min(len(OPEN_TAG) - 1, len(self._buffer))
                if keep and OPEN_TAG.startswith(self._buffer[-keep:]):
                    visible.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                else:
                    visible.append(self._buffer)
                    self._buffer = ""
                break

            visible.append(self._buffer[:open_at])
            self._buffer = self._buffer[open_at + len(OPEN_TAG) :]
            self._inside = True

        return "".join(visible)

    def flush(self) -> str:
        if self._inside:
            self._buffer = ""
            return ""
        leftover = self._buffer
        self._buffer = ""
        return leftover
