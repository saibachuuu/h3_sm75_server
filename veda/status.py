"""VEDA runtime diagnostics are written only to the console log."""

from __future__ import annotations

import logging

_LOG = logging.getLogger('Star7-H3-VEDA')


class NodeStatus:
    """Logs runtime status without adding a node text widget."""

    def __init__(self, node_id: str | None):
        del node_id
        self._last = None

    def show(self, text: str, level: int = logging.INFO) -> None:
        plain = text.replace(' · ', ' | ').strip()
        if self._last == (level, plain):
            return
        self._last = (level, plain)
        for line in plain.splitlines():
            if line.strip():
                _LOG.log(level, '[Star7 H3 VEDA] %s', line)

    def warn(self, text: str) -> None:
        self.show(text, logging.WARNING)
