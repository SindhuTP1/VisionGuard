"""Read individual members of a large remote .zip using HTTP range requests.

Used to pull only the HADM validation split (a few hundred MB) out of the 8.8 GB
archive. ``zipfile`` only needs a seekable file object; this one fetches byte
ranges on demand, with a small block cache.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20, timeout: float = 60.0) -> None:
        self.url, self.block, self.timeout = url, block, timeout
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 206:
                raise IOError("Server does not support range requests")
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except Exception:  # noqa: BLE001 - retry transient network errors
                if attempt == 3:
                    raise
        raise AssertionError("unreachable")

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        # Large reads go straight through; small ones use the block cache.
        if n >= self.block:
            data = self._fetch(self.pos, self.pos + n - 1)
        else:
            out = bytearray()
            p = self.pos
            while len(out) < n:
                b = p // self.block
                if b not in self._cache:
                    if len(self._cache) > 64:
                        self._cache.clear()
                    self._cache[b] = self._fetch(b * self.block, min(self.size, (b + 1) * self.block) - 1)
                chunk = self._cache[b][p - b * self.block:]
                take = chunk[: n - len(out)]
                out += take
                p += len(take)
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpRangeFile(url))
