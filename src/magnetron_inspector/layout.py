from __future__ import annotations

import struct
from dataclasses import dataclass, field

from magnetron._magnetron_bindings import _SNAPSHOT_TBLOB_ALIGN

from .snapshot import Snapshot

# Mirrors magnetron/core/mag_io_snapshot_layout.h
HEADER_FORMAT = '<III QQ QQ'  # magic, version, aux, meta offset, meta span, blob offset, blob span (always little endian)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = int.from_bytes(b'MAG!', 'little')
FILE_BLOB_ALIGN = 0x1000
TENSOR_BLOB_ALIGN = int(_SNAPSHOT_TBLOB_ALIGN)
AUX_BIG_ENDIAN = 1 << 0

KINDS = ('header', 'metadata', 'padding', 'tensor', 'gap', 'trailing')


@dataclass(frozen=True, slots=True)
class Header:
    magic: int
    version: int
    aux: int
    meta_offset: int
    meta_span: int
    blob_offset: int
    blob_span: int

    @property
    def magic_text(self) -> str:
        return self.magic.to_bytes(4, 'little').decode('latin-1')

    @property
    def big_endian(self) -> bool:
        return bool(self.aux & AUX_BIG_ENDIAN)


@dataclass(slots=True)
class Region:
    kind: str
    name: str
    start: int  # Absolute file offset
    end: int
    dtype: str = ''
    issues: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class Layout:
    header: Header
    file_size: int
    regions: list[Region]
    issues: list[str]

    @property
    def payload_bytes(self) -> int:
        return sum(r.size for r in self.regions if r.kind == 'tensor')

    @property
    def padding_bytes(self) -> int:
        return sum(r.size for r in self.regions if r.kind in ('padding', 'gap'))


def read_header(path: str) -> Header:
    with open(path, 'rb') as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f'File is {len(raw)} bytes, smaller than the {HEADER_SIZE} byte header')
    return Header(*struct.unpack(HEADER_FORMAT, raw))


def build_layout(snap: Snapshot) -> Layout:
    """Every byte of the file attributed to a region, in file order, with layout rule violations attached."""
    hdr = read_header(snap.path)
    issues: list[str] = []
    regions: list[Region] = [Region('header', 'file header', 0, HEADER_SIZE)]
    if hdr.magic != MAGIC:
        regions[0].issues.append(f'magic is {hdr.magic_text!r}, expected MAG!')
    if hdr.aux & ~AUX_BIG_ENDIAN:
        regions[0].issues.append(f'reserved aux bits set: 0x{hdr.aux:08X}')

    meta = Region('metadata', 'metadata JSON', hdr.meta_offset, hdr.meta_offset + hdr.meta_span)
    meta_len = len(snap.metadata_raw.encode('utf-8'))
    if meta_len != hdr.meta_span:
        meta.issues.append(f'header says {hdr.meta_span:,} bytes but the document is {meta_len:,}')
    if hdr.meta_offset < HEADER_SIZE:
        meta.issues.append('overlaps the header')
    regions.append(meta)

    if hdr.blob_offset > meta.end:
        regions.append(Region('padding', 'pad to data section', meta.end, hdr.blob_offset))
    elif hdr.blob_offset < meta.end:
        issues.append('data section overlaps the metadata')
    blob_issue = []
    if hdr.blob_offset % FILE_BLOB_ALIGN:
        blob_issue.append(f'data section starts at 0x{hdr.blob_offset:X}, not {FILE_BLOB_ALIGN}-byte aligned')
    if hdr.blob_span != snap.blob_numbytes:
        blob_issue.append(f'header blob span {hdr.blob_span:,} differs from reader {snap.blob_numbytes:,}')
    issues.extend(blob_issue)

    cursor = 0  # Relative to the data section
    for info in sorted(snap.tensors.values(), key=lambda t: (t.offset, t.nbytes)):
        r = Region('tensor', info.name, hdr.blob_offset + info.offset, hdr.blob_offset + info.offset + info.nbytes, info.dtype)
        gap = info.offset - cursor
        if gap < 0:
            r.issues.append(f'overlaps the previous region by {-gap:,} bytes')
        elif gap > 0:
            kind = 'padding' if gap < TENSOR_BLOB_ALIGN else 'gap'
            pad = Region(kind, 'alignment padding' if kind == 'padding' else 'unattributed bytes', hdr.blob_offset + cursor, r.start)
            if kind == 'gap':
                pad.issues.append(f'{gap:,} bytes belong to no tensor, more than the {TENSOR_BLOB_ALIGN} byte alignment explains')
            regions.append(pad)
        if info.nbytes and info.offset % TENSOR_BLOB_ALIGN:
            r.issues.append(f'offset 0x{info.offset:X} is not {TENSOR_BLOB_ALIGN}-byte aligned')
        if info.offset + info.nbytes > hdr.blob_span:
            r.issues.append('runs past the end of the data section')
        regions.append(r)
        cursor = max(cursor, info.offset + info.nbytes)

    blob_end = hdr.blob_offset + hdr.blob_span
    if hdr.blob_span > cursor:
        tail = Region('padding', 'tail padding', hdr.blob_offset + cursor, blob_end)
        if hdr.blob_span - cursor >= TENSOR_BLOB_ALIGN:
            tail.kind = 'gap'
            tail.issues.append('more unused bytes than alignment explains')
        regions.append(tail)
    if snap.file_size > blob_end:
        regions.append(Region('trailing', 'bytes after data section', blob_end, snap.file_size, issues=['not covered by the header']))
    elif snap.file_size < blob_end:
        issues.append(f'file is {blob_end - snap.file_size:,} bytes shorter than the header claims')

    return Layout(hdr, snap.file_size, regions, issues + [f'{r.name}: {i}' for r in regions for i in r.issues])
