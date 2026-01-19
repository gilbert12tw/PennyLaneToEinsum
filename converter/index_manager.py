from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import string


_ASCII_SYMBOLS = string.ascii_letters
_UNICODE_CACHE: List[str] = []
_UNICODE_SCAN_POS = 0x00A1


def _get_unicode_symbol(index: int) -> str:
    global _UNICODE_SCAN_POS
    while len(_UNICODE_CACHE) <= index:
        if _UNICODE_SCAN_POS > 0x10FFFF:
            raise ValueError("Exhausted Unicode symbols for einsum indices")
        ch = chr(_UNICODE_SCAN_POS)
        _UNICODE_SCAN_POS += 1
        if ch.isalpha():
            _UNICODE_CACHE.append(ch)
    return _UNICODE_CACHE[index]


def _index_from_int(value: int) -> str:
    if value < len(_ASCII_SYMBOLS):
        return _ASCII_SYMBOLS[value]
    return _get_unicode_symbol(value - len(_ASCII_SYMBOLS))


@dataclass
class IndexManager:
    n_qubits: int
    counter: int = 0
    qubit_indices: Dict[int, str] = field(default_factory=dict)

    def init_qubits(self) -> Dict[int, str]:
        self.qubit_indices = {}
        for q in range(self.n_qubits):
            self.qubit_indices[q] = self.fresh_index()
        return dict(self.qubit_indices)

    def fresh_index(self) -> str:
        idx = _index_from_int(self.counter)
        self.counter += 1
        return idx

    def get(self, qubit: int) -> str:
        return self.qubit_indices[qubit]

    def set(self, qubit: int, index: str) -> None:
        self.qubit_indices[qubit] = index

    def snapshot(self) -> Dict[int, str]:
        return dict(self.qubit_indices)

