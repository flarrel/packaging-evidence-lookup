"""Bounded, atomic fetches for the project's public research inputs."""

from collections.abc import Sequence
import hashlib
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


def download(
    url: str,
    target: Path,
    limit: int,
    checksums: Sequence[tuple[str, str]] = (),
) -> Path:
    def verify(path: Path) -> bool:
        for algorithm, expected in checksums:
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, algorithm).hexdigest() != expected:
                    return False
        return 0 < path.stat().st_size <= limit

    if target.is_file() and verify(target):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "PackagingEvidenceLookup/1.0"})
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with urlopen(request, timeout=45) as response, staging.open("wb") as output:
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > limit:
                    raise ValueError(f"Download exceeded its byte limit: {url}")
                output.write(chunk)
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != total:
                raise ValueError(f"Incomplete download: {url}")
        if not verify(staging):
            raise ValueError(f"Download checksum did not match: {url}")
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
    return target
