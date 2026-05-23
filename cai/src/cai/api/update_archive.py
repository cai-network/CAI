# SPDX-FileCopyrightText: 2025 cai Technologies Ltd
# SPDX-FileCopyrightText: 2026 CAI contributors
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Iterable
from http import HTTPStatus
from pathlib import Path

from fastapi.responses import Response, StreamingResponse


CAI_UPDATE_ARCHIVE_STREAM_CHUNK_BYTES = 1024 * 1024


def resolve_cai_repo_root() -> Path:
    configured = str(
        os.getenv("CAI_REPO_ROOT")
        or os.getenv("CAI_RUNTIME_REPO")
        or ""
    ).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def stream_cai_update_archive_response(
    archive_path: Path,
    *,
    range_header: str | None,
) -> Response | StreamingResponse:
    resolved = archive_path.expanduser().resolve()
    size = resolved.stat().st_size
    filename = resolved.name.replace('"', "")
    base_headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'attachment; filename="{filename}"',
    }

    try:
        requested_range = parse_cai_update_archive_range(range_header, size=size)
    except ValueError:
        return Response(
            status_code=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={**base_headers, "Content-Range": f"bytes */{size}"},
        )

    if requested_range is None:
        start = 0
        end = size - 1
        return StreamingResponse(
            iter_cai_update_archive_range(resolved, start=start, end=end),
            media_type="application/zip",
            headers={**base_headers, "Content-Length": str(size)},
        )

    start, end = requested_range
    content_length = max(0, end - start + 1)
    return StreamingResponse(
        iter_cai_update_archive_range(resolved, start=start, end=end),
        status_code=HTTPStatus.PARTIAL_CONTENT,
        media_type="application/zip",
        headers={
            **base_headers,
            "Content-Length": str(content_length),
            "Content-Range": f"bytes {start}-{end}/{size}",
        },
    )


def parse_cai_update_archive_range(
    range_header: str | None,
    *,
    size: int,
) -> tuple[int, int] | None:
    header = str(range_header or "").strip()
    if not header:
        return None
    if size <= 0:
        raise ValueError("Cannot serve ranges for an empty archive.")
    if not header.lower().startswith("bytes="):
        raise ValueError("Unsupported range unit.")
    spec = header.split("=", 1)[1].strip()
    if not spec or "," in spec:
        raise ValueError("Only a single byte range is supported.")
    start_text, separator, end_text = spec.partition("-")
    if separator != "-":
        raise ValueError("Invalid byte range.")

    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("Invalid suffix byte range.")
        start = max(size - suffix_length, 0)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Requested range is not satisfiable.")
    return start, min(end, size - 1)


def iter_cai_update_archive_range(
    archive_path: Path,
    *,
    start: int,
    end: int,
) -> Iterable[bytes]:
    remaining = max(0, end - start + 1)
    with archive_path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            chunk = handle.read(
                min(CAI_UPDATE_ARCHIVE_STREAM_CHUNK_BYTES, remaining)
            )
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
