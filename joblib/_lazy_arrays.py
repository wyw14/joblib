"""Explicit chunked persistence format and lazy numpy array proxies.

This module implements an opt-in on-disk format where non-object numpy
arrays are stored as a sequence of independently compressed, fixed upper
size contiguous chunks. The format is fully self-describing:

* an explicit magic number and format version,
* the array shape, dtype (preserving byte order and structured dtypes)
  and C/F storage order,
* a chunk index recording the (offset, encoded length, decoded length)
  triple of every compressed chunk,
* length checks on both the encoded and the decoded side so that corrupt
  or oversized declarations are rejected before any buffer is allocated.

When a file written in this format is loaded with ``lazy_arrays=True``,
array data is not materialized: :class:`LazyArrayProxy` objects are built
instead and only the compressed chunks intersecting a given indexing
operation are read, decompressed and cached in a byte-budgeted LRU.

Object dtypes are never written in this format: they keep using the
legacy eager path, and attempting to create a chunked payload for an
object dtype raises an explicit error instead of serializing Python
object pointers as raw numeric values.
"""

# Author: joblib developers
# License: BSD Style, 3 clauses.

import contextlib
import io
import os
import pickle
import struct
import threading
import zlib
from collections import OrderedDict

from .numpy_pickle_utils import (
    _ensure_native_byte_order,
    _read_bytes,
    _reconstruct,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


###############################################################################
# Format constants
###############################################################################

# Magic number of the explicit chunked format. The trailing digit carries
# the on-disk layout generation; _FORMAT_VERSION is additionally stored in
# every header for forward/backward compatibility checks.
_CHUNKED_MAGIC = b"JLC1"
_FORMAT_VERSION = 1

# Per-chunk compression codecs stored in the header.
_COMPRESSOR_RAW = 0
_COMPRESSOR_ZLIB = 1
_COMPRESSOR_IDS = {_COMPRESSOR_RAW, _COMPRESSOR_ZLIB}
_COMPRESSOR_NAMES = {"none": _COMPRESSOR_RAW, "zlib": _COMPRESSOR_ZLIB}
_COMPRESSOR_NAMES_INV = {v: k for k, v in _COMPRESSOR_NAMES.items()}

# Hard sanity bound on header size: a corrupt header length declaration is
# rejected before the header buffer itself is allocated.
_MAX_HEADER_BYTES = 64 * 1024**2

# Default size (in raw bytes) of a chunk. The number of elements per chunk
# is derived from the array itemsize, giving a fixed upper byte size.
DEFAULT_CHUNK_BYTES = 8 * 1024**2

# Default resource limits applied when loading with lazy_arrays=True.
DEFAULT_LAZY_CACHE_BYTES = 128 * 1024**2
DEFAULT_LAZY_MAX_CHUNK_BYTES = 256 * 1024**2

# Single source of truth for the configurable lazy loading limits. Always
# read these through get_lazy_options() so runtime updates propagate.
_LAZY_OPTIONS = {
    "cache_bytes": DEFAULT_LAZY_CACHE_BYTES,
    "max_chunk_bytes": DEFAULT_LAZY_MAX_CHUNK_BYTES,
}


class LazyArrayError(ValueError):
    """Error raised by the chunked persistence format or lazy proxies."""


def get_lazy_options():
    """Return a copy of the current lazy loading resource limits."""
    return dict(_LAZY_OPTIONS)


def _validate_positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            "'{}' must be a strictly positive integer, got {!r}.".format(name, value)
        )
    return value


def set_lazy_options(cache_bytes=None, max_chunk_bytes=None):
    """Configure the default lazy loading resource limits.

    Parameters
    ----------
    cache_bytes: int, optional
        Default maximum number of *decoded* bytes retained by the LRU
        cache of each lazy source. Individual ``load`` calls may override
        it through the ``cache_bytes`` argument.
    max_chunk_bytes: int, optional
        Hard limit on the decoded size of a single chunk. A chunk whose
        index declaration exceeds this size is rejected, before any
        output buffer is allocated.
    """
    if cache_bytes is not None:
        _validate_positive_int(cache_bytes, "cache_bytes")
        _LAZY_OPTIONS["cache_bytes"] = cache_bytes
    if max_chunk_bytes is not None:
        _validate_positive_int(max_chunk_bytes, "max_chunk_bytes")
        _LAZY_OPTIONS["max_chunk_bytes"] = max_chunk_bytes


@contextlib.contextmanager
def lazy_options(cache_bytes=None, max_chunk_bytes=None):
    """Context manager temporarily overriding lazy loading limits."""
    previous = get_lazy_options()
    set_lazy_options(cache_bytes=cache_bytes, max_chunk_bytes=max_chunk_bytes)
    try:
        yield
    finally:
        _LAZY_OPTIONS["cache_bytes"] = previous["cache_bytes"]
        _LAZY_OPTIONS["max_chunk_bytes"] = previous["max_chunk_bytes"]


###############################################################################
# Header (de)serialization
###############################################################################


class _ChunkedHeader:
    """Parsed description of a chunked array payload."""

    __slots__ = (
        "version",
        "compressor_id",
        "order",
        "shape",
        "dtype",
        "chunk_elements",
        "n_chunks",
        "index",
        "total_bytes",
        "data_start_rel",
        "footer_rel",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs[slot])

    @property
    def n_elements(self):
        if len(self.shape) == 0:
            return 1
        count = np.int64(1)
        for dim in self.shape:
            count *= np.int64(dim)
        return int(count)

    def chunk_element_count(self, chunk_id):
        """Number of raw elements stored in chunk ``chunk_id``."""
        if chunk_id == self.n_chunks - 1:
            remainder = self.n_elements - (self.n_chunks - 1) * self.chunk_elements
            return remainder
        return self.chunk_elements

    def snapshot(self):
        """Picklable identity snapshot used to verify reopened files."""
        return (
            self.version,
            self.compressor_id,
            self.order,
            tuple(self.shape),
            self.dtype.str,
            self.dtype.descr,
            self.chunk_elements,
            self.n_chunks,
            tuple(self.index),
            self.total_bytes,
        )


def _encode_index(index):
    return b"".join(
        struct.pack("<QQQ", offset, encoded_len, decoded_len)
        for offset, encoded_len, decoded_len in index
    )


def _encode_header(compressor_id, order, shape, dtype, chunk_elements, index):
    """Build the header bytes. Returns (header_bytes, index_offset)."""
    dtype_blob = pickle.dumps(dtype, protocol=pickle.HIGHEST_PROTOCOL)
    fixed = struct.pack(
        "<HBBII",
        _FORMAT_VERSION,
        compressor_id,
        ord(order),
        len(shape),
        len(index),
    )
    shape_bytes = b"".join(struct.pack("<q", int(dim)) for dim in shape)
    params = struct.pack("<QII", int(chunk_elements), len(dtype_blob), len(index))
    index_offset = len(fixed) + len(shape_bytes) + len(params) + len(dtype_blob)
    return (
        fixed + shape_bytes + params + dtype_blob + _encode_index(index),
        index_offset,
    )


def _read_exact(fh, size, what):
    try:
        return _read_bytes(fh, size, what)
    except ValueError as exc:  # ran out of data -> treat as corruption
        raise LazyArrayError(
            "Truncated chunked array payload while reading {}: {}".format(what, exc)
        ) from exc


def parse_chunked_header(fh, base=None, max_chunk_bytes=None):
    """Parse a chunked payload header located at the current file position.

    The file position is left at the beginning of the chunk data region.
    """
    if base is None:
        base = fh.tell()

    magic = _read_exact(fh, len(_CHUNKED_MAGIC), "magic number")
    if magic != _CHUNKED_MAGIC:
        raise LazyArrayError(
            "Invalid chunked array magic number {!r}, expected {!r}.".format(
                magic, _CHUNKED_MAGIC
            )
        )

    (header_length,) = struct.unpack("<I", _read_exact(fh, 4, "header length"))
    if header_length == 0 or header_length > _MAX_HEADER_BYTES:
        # Reject before allocating the header buffer.
        raise LazyArrayError(
            "Declared chunked header length {} is out of bounds (1, {}).".format(
                header_length, _MAX_HEADER_BYTES
            )
        )
    header_bytes = _read_exact(fh, header_length, "header")
    data_start_rel = 8 + header_length

    stream = io.BytesIO(header_bytes)

    def unpack(fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, _read_exact(stream, size, "header field"))

    version, compressor_id, order_byte, ndim, n_chunks = unpack("<HBBII")
    if version != _FORMAT_VERSION:
        raise LazyArrayError(
            "Unsupported chunked array format version {}. Only version {} "
            "is supported.".format(version, _FORMAT_VERSION)
        )
    if compressor_id not in _COMPRESSOR_IDS:
        raise LazyArrayError(
            "Unknown chunk compressor id {} in chunked array header.".format(
                compressor_id
            )
        )
    order = chr(order_byte)
    if order not in ("C", "F"):
        raise LazyArrayError(
            "Invalid storage order {!r} in chunked array header.".format(order)
        )
    if ndim > 64:
        raise LazyArrayError(
            "Unreasonable ndim {} in chunked array header.".format(ndim)
        )
    shape = tuple(unpack("<q")[0] for _ in range(ndim))
    if any(dim < 0 for dim in shape):
        raise LazyArrayError("Negative shape dimension in chunked array header.")

    chunk_elements, dtype_length, index_count = unpack("<QII")
    if dtype_length == 0 or dtype_length > header_length:
        raise LazyArrayError("Invalid dtype declaration length in chunked header.")
    dtype_blob = _read_exact(stream, dtype_length, "dtype")
    try:
        dtype = pickle.loads(dtype_blob)
    except Exception as exc:
        raise LazyArrayError(
            "Could not decode dtype from chunked array header."
        ) from exc
    if not isinstance(dtype, np.dtype):
        raise LazyArrayError("Chunked array header did not decode to a numpy dtype.")
    if dtype.hasobject:
        # Defensive check: object dtypes must never reach the raw chunk path.
        raise LazyArrayError(
            "Object dtypes are not supported by the chunked array format."
        )
    if index_count != n_chunks:
        raise LazyArrayError(
            "Chunk index count {} does not match declared chunk count {}.".format(
                index_count, n_chunks
            )
        )
    if n_chunks > 0 and chunk_elements < 1:
        raise LazyArrayError("chunk_elements must be strictly positive.")

    index = []
    for _ in range(n_chunks):
        offset, encoded_len, decoded_len = unpack("<QQQ")
        index.append((int(offset), int(encoded_len), int(decoded_len)))

    # Cross-validate shape-derived sizes against the index.
    if len(shape) == 0:
        n_elements = 1
    else:
        n_elements = 1
        for dim in shape:
            n_elements *= int(dim)
    if n_elements > 0:
        expected_chunks = (n_elements + chunk_elements - 1) // chunk_elements
    else:
        expected_chunks = 0
    if expected_chunks != n_chunks:
        raise LazyArrayError(
            "Chunk count {} is inconsistent with shape {} and chunk size "
            "{} (expected {} chunks).".format(
                n_chunks, shape, chunk_elements, expected_chunks
            )
        )

    # Validate the index geometry: offsets must be contiguous, located in the
    # data region, and declared decoded lengths must match the dtype/shape.
    expected_offset = data_start_rel
    itemsize = dtype.itemsize
    for chunk_id, (offset, encoded_len, decoded_len) in enumerate(index):
        if offset != expected_offset:
            raise LazyArrayError(
                "Chunk {} offset {} does not match expected position {}.".format(
                    chunk_id, offset, expected_offset
                )
            )
        if encoded_len <= 0:
            raise LazyArrayError(
                "Chunk {} declares a non-positive encoded length {}.".format(
                    chunk_id, encoded_len
                )
            )
        expected_elements = (
            n_elements - chunk_id * chunk_elements
            if chunk_id == n_chunks - 1
            else chunk_elements
        )
        expected_decoded = expected_elements * itemsize
        if decoded_len != expected_decoded:
            raise LazyArrayError(
                "Chunk {} declares {} decoded bytes but shape/dtype require "
                "{} bytes.".format(chunk_id, decoded_len, expected_decoded)
            )
        if max_chunk_bytes is not None and decoded_len > max_chunk_bytes:
            # Checked before any chunk buffer allocation.
            raise LazyArrayError(
                "Chunk {} decoded size {} bytes exceeds the configured "
                "max_chunk_bytes limit of {} bytes.".format(
                    chunk_id, decoded_len, max_chunk_bytes
                )
            )
        expected_offset = offset + encoded_len
    footer_rel = expected_offset

    # The footer stores the total decoded byte count. It is located right
    # after the chunk data region, seek there explicitly and restore the
    # stream position at the start of the data region afterwards.
    fh.seek(base + footer_rel)
    (total_bytes,) = struct.unpack("<Q", _read_exact(fh, 8, "chunked footer"))
    if total_bytes != n_elements * itemsize:
        raise LazyArrayError(
            "Chunked footer declares {} total bytes but shape/dtype require "
            "{} bytes.".format(total_bytes, n_elements * itemsize)
        )
    if sum(decoded for _, _, decoded in index) != total_bytes:
        raise LazyArrayError(
            "Sum of chunk decoded lengths does not match chunked footer."
        )

    # When the underlying stream is seekable, verify that every declared
    # encoded region fits inside the file before any chunk is read.
    if hasattr(fh, "seek") and hasattr(fh, "tell"):
        try:
            current = fh.tell()
            fh.seek(0, os.SEEK_END)
            file_size = fh.tell()
            fh.seek(current)
        except (OSError, io.UnsupportedOperation):
            file_size = None
        if file_size is not None:
            payload_end = base + footer_rel + 8
            if payload_end > file_size:
                raise LazyArrayError(
                    "Chunked payload extends to byte {} but the file only "
                    "contains {} bytes.".format(payload_end, file_size)
                )
            for offset, encoded_len, _ in index:
                if base + offset + encoded_len > base + footer_rel:
                    raise LazyArrayError(
                        "Chunk index entry extends past the chunk data region."
                    )

    # Leave the stream right after the header (start of the data region).
    fh.seek(base + data_start_rel)

    return _ChunkedHeader(
        version=version,
        compressor_id=compressor_id,
        order=order,
        shape=shape,
        dtype=dtype,
        chunk_elements=int(chunk_elements),
        n_chunks=n_chunks,
        index=index,
        total_bytes=int(total_bytes),
        data_start_rel=data_start_rel,
        footer_rel=footer_rel,
    )


###############################################################################
# Byte-budgeted LRU cache and shared file sources
###############################################################################


class _ByteBudgetLRU:
    """LRU cache whose eviction policy is driven by retained *bytes*."""

    def __init__(self, byte_budget):
        _validate_positive_int(byte_budget, "byte_budget")
        self._byte_budget = byte_budget
        self._items = OrderedDict()
        self._retained_bytes = 0

    @property
    def byte_budget(self):
        return self._byte_budget

    @property
    def retained_bytes(self):
        return self._retained_bytes

    def __len__(self):
        return len(self._items)

    def get(self, key):
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key, value):
        existing = self._items.get(key)
        if existing is not None:
            self._retained_bytes -= len(existing)
            self._items.move_to_end(key)
        self._items[key] = value
        self._retained_bytes += len(value)
        # Evict least-recently-used entries until the budget is respected.
        # A single freshly inserted entry larger than the budget is kept on
        # its own: the hard max_chunk_bytes cap bounds its absolute size.
        while self._retained_bytes > self._byte_budget and len(self._items) > 1:
            _, dropped = self._items.popitem(last=False)
            self._retained_bytes -= len(dropped)

    def clear(self):
        self._items.clear()
        self._retained_bytes = 0


# Process-wide registry of sources opened from a filename, so that several
# proxies (or several load calls) pointing at the same file share one handle
# and one LRU cache.
_SOURCE_REGISTRY = {}
_SOURCE_REGISTRY_LOCK = threading.Lock()


def _file_identity(fobj):
    try:
        stat = os.fstat(fobj.fileno())
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    except (OSError, AttributeError, io.UnsupportedOperation):
        return None


class _LazyFileSource:
    """A seekable file handle shared by one or several lazy array proxies."""

    def __init__(
        self,
        file_handle,
        filename=None,
        owns_handle=False,
        cache_bytes=None,
        max_chunk_bytes=None,
        identity=None,
    ):
        options = get_lazy_options()
        self._file_handle = file_handle
        self.filename = filename
        self.owns_handle = owns_handle
        self.cache_bytes = (
            _validate_positive_int(cache_bytes, "cache_bytes")
            if cache_bytes is not None
            else options["cache_bytes"]
        )
        self.max_chunk_bytes = (
            _validate_positive_int(max_chunk_bytes, "max_chunk_bytes")
            if max_chunk_bytes is not None
            else options["max_chunk_bytes"]
        )
        self.identity = identity
        self._cache = _ByteBudgetLRU(self.cache_bytes)
        self._lock = threading.RLock()
        self._refcount = 0
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def acquire(self):
        with self._lock:
            if self._closed:
                raise LazyArrayError("Cannot use a closed lazy array source.")
            self._refcount += 1

    def release(self):
        with self._lock:
            self._refcount -= 1
            if self._refcount <= 0:
                self._refcount = 0
                self.close()

    def close(self):
        """Explicitly close the source.

        Only file handles owned by the source (i.e. opened by joblib from a
        filename) are closed; user-provided file objects stay open unless
        they are explicitly closed through their own API.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            if self.owns_handle and not self._file_handle.closed:
                self._file_handle.close()
            if self.filename is not None:
                with _SOURCE_REGISTRY_LOCK:
                    _SOURCE_REGISTRY.pop(os.path.abspath(self.filename), None)

    def read_chunk(self, base, header, chunk_id):
        """Read and decode one chunk, returning a 1-D numpy array."""
        with self._lock:
            if self._closed:
                raise LazyArrayError("Cannot read from a closed lazy array source.")
            fh = self._file_handle
            offset, encoded_len, decoded_len = header.index[chunk_id]
            expected_elements = header.chunk_element_count(chunk_id)
            expected_bytes = expected_elements * header.dtype.itemsize

            # All size checks happen before any allocation.
            if decoded_len != expected_bytes:
                raise LazyArrayError(
                    "Corrupt chunk index: chunk {} declares {} decoded bytes, "
                    "expected {}.".format(chunk_id, decoded_len, expected_bytes)
                )
            if decoded_len > self.max_chunk_bytes:
                raise LazyArrayError(
                    "Chunk {} decoded size {} bytes exceeds max_chunk_bytes "
                    "limit of {} bytes.".format(
                        chunk_id, decoded_len, self.max_chunk_bytes
                    )
                )
            if encoded_len <= 0:
                raise LazyArrayError(
                    "Corrupt chunk index: non-positive encoded length for "
                    "chunk {}.".format(chunk_id)
                )

            cache_key = (base, chunk_id)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return np.frombuffer(
                    cached, dtype=header.dtype, count=expected_elements
                )

            fh.seek(base + offset)
            encoded = _read_exact(
                fh, encoded_len, "chunk {} encoded data".format(chunk_id)
            )
            if header.compressor_id == _COMPRESSOR_RAW:
                decoded = encoded
            else:
                try:
                    decoded = zlib.decompress(encoded)
                except zlib.error as exc:
                    # Never substitute a zero-filled block on failure.
                    raise LazyArrayError(
                        "Failed to decompress chunk {}: {}".format(chunk_id, exc)
                    ) from exc
            if len(decoded) != decoded_len:
                raise LazyArrayError(
                    "Chunk {} decoded to {} bytes but its index entry declares "
                    "{} bytes.".format(chunk_id, len(decoded), decoded_len)
                )

            self._cache.put(cache_key, decoded)
            array = np.frombuffer(decoded, dtype=header.dtype, count=expected_elements)
            if array.shape[0] != expected_elements:
                raise LazyArrayError(
                    "Chunk {} element count mismatch.".format(chunk_id)
                )
            return array

    def cache_info(self):
        with self._lock:
            return {
                "entries": len(self._cache),
                "retained_bytes": self._cache.retained_bytes,
                "byte_budget": self._cache.byte_budget,
            }


def _open_source_from_handle(
    file_handle, filename, owns_handle, cache_bytes, max_chunk_bytes
):
    identity = _file_identity(file_handle) if filename is not None else None
    source = _LazyFileSource(
        file_handle,
        filename=filename,
        owns_handle=owns_handle,
        cache_bytes=cache_bytes,
        max_chunk_bytes=max_chunk_bytes,
        identity=identity,
    )
    if filename is not None and owns_handle:
        with _SOURCE_REGISTRY_LOCK:
            _SOURCE_REGISTRY[os.path.abspath(filename)] = source
    return source


def _get_or_create_registered_source(filename, cache_bytes=None, max_chunk_bytes=None):
    """Return a registered source for ``filename``, reusing it when possible.

    No reference is acquired here: references are owned by the proxies built
    from the source. A fresh handle is opened when no source exists or the
    file identity changed.
    """
    absolute = os.path.abspath(filename)
    with _SOURCE_REGISTRY_LOCK:
        existing = _SOURCE_REGISTRY.get(absolute)
    if existing is not None and not existing.closed:
        with open(absolute, "rb") as probe:
            current_identity = _file_identity(probe)
        if current_identity == existing.identity:
            return existing
        with _SOURCE_REGISTRY_LOCK:
            _SOURCE_REGISTRY.pop(absolute, None)

    file_handle = open(absolute, "rb")
    try:
        return _open_source_from_handle(
            file_handle,
            absolute,
            owns_handle=True,
            cache_bytes=cache_bytes,
            max_chunk_bytes=max_chunk_bytes,
        )
    except BaseException:
        file_handle.close()
        raise


def _source_for_filename(filename, cache_bytes=None, max_chunk_bytes=None):
    """Open (or reuse) a registered source, acquiring one proxy reference."""
    source = _get_or_create_registered_source(
        filename, cache_bytes=cache_bytes, max_chunk_bytes=max_chunk_bytes
    )
    source.acquire()
    return source


###############################################################################
# Lazy array proxy
###############################################################################


def _expand_key(key, ndim):
    """Expand Ellipsis/newaxis and return one entry per source axis marker."""
    if not isinstance(key, tuple):
        key = (key,)
    if key.count(Ellipsis) > 1:
        raise IndexError("Only one Ellipsis is allowed in an index.")
    explicit = sum(1 for entry in key if entry is not None and entry is not Ellipsis)
    if explicit > ndim:
        raise IndexError(
            "Too many indices for array: array is {}-dimensional but {} "
            "were indexed.".format(ndim, explicit)
        )
    fill = ndim - explicit
    expanded = []
    for entry in key:
        if entry is Ellipsis:
            expanded.extend([slice(None)] * fill)
        else:
            expanded.append(entry)
    while sum(1 for entry in expanded if entry is not None) < ndim:
        expanded.append(slice(None))
    if sum(1 for entry in expanded if entry is not None) != ndim:
        raise IndexError("Too many indices for array.")
    return expanded


class LazyArrayProxy:
    """Proxy to a chunked numpy array that reads chunks on demand.

    The proxy supports basic indexing (integers, slices including negative
    strides, ``Ellipsis`` and ``np.newaxis``/``None``). Only the compressed
    chunks intersecting a selection are read; decoded chunks are shared in a
    byte-budgeted LRU cache owned by the underlying :class:`_LazyFileSource`.
    """

    def __init__(
        self, source, base, header, ensure_native_byte_order=False, subclass=None
    ):
        self._source = source
        self._base = int(base)
        self._header = header
        self._ensure_native = bool(ensure_native_byte_order)
        self._subclass = subclass
        self._closed = False
        source.acquire()

    # -- introspection -----------------------------------------------------

    @property
    def shape(self):
        return self._header.shape

    @property
    def ndim(self):
        return len(self._header.shape)

    @property
    def dtype(self):
        return self._header.dtype

    @property
    def size(self):
        return self._header.n_elements

    @property
    def nbytes(self):
        return self._header.total_bytes

    @property
    def order(self):
        return self._header.order

    @property
    def source(self):
        return self._source

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __repr__(self):
        return "LazyArrayProxy(shape={}, dtype={}, order={!r}, file={!r})".format(
            self.shape,
            self.dtype,
            self.order,
            getattr(self._source, "filename", None),
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self):
        """Release this proxy's reference on the shared source."""
        if self._closed:
            return
        self._closed = True
        self._source.release()

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - best effort finalization
            pass

    def _check_open(self):
        if self._closed or self._source.closed:
            raise LazyArrayError("Operation on a closed lazy array proxy.")

    # -- pickle support: reopen the file and verify its identity -----------

    def __reduce__(self):
        if self._source.filename is None or not self._source.owns_handle:
            raise TypeError(
                "Lazy array proxies backed by a user-provided file object "
                "cannot be pickled: only proxies loaded from a filename can "
                "reopen their source file."
            )
        return (
            _rebuild_lazy_proxy,
            (
                self._source.filename,
                self._source.identity,
                self._base,
                self._header.snapshot(),
                self._source.cache_bytes,
                self._source.max_chunk_bytes,
                self._ensure_native,
            ),
        )

    # -- materialization ---------------------------------------------------

    def _finalize(self, array):
        if self._ensure_native:
            array = _ensure_native_byte_order(array)
        if (
            self._subclass is not None
            and hasattr(self._subclass, "__array_prepare__")
            and self._subclass not in (np.ndarray, getattr(np, "memmap", None))
        ):
            new_array = _reconstruct(self._subclass, (0,), "b")
            return new_array.__array_prepare__(array)
        return array

    def __array__(self, dtype=None):
        """Explicit full materialization of the array."""
        self._check_open()
        header = self._header
        if header.n_elements == 0:
            array = np.empty(0, dtype=header.dtype)
        else:
            array = np.empty(header.n_elements, dtype=header.dtype)
            for chunk_id in range(header.n_chunks):
                start = chunk_id * header.chunk_elements
                end = min(header.n_elements, start + header.chunk_elements)
                chunk = self._source.read_chunk(self._base, header, chunk_id)
                array[start:end] = chunk
        if header.order == "F":
            array = array.reshape(header.shape[::-1]).transpose()
        else:
            array = array.reshape(header.shape)
        array = self._finalize(array)
        if dtype is not None and np.dtype(dtype) != array.dtype:
            array = array.astype(dtype)
        return array

    # -- indexing ----------------------------------------------------------

    def __getitem__(self, key):
        self._check_open()
        header = self._header
        shape = header.shape
        ndim = len(shape)

        if ndim == 0:
            # A 0-d array holds a single element.
            return self.__array__()[key]

        expanded = _expand_key(key, ndim)

        axis_specs = []  # entries: ("int", value) or ("slice", start, stop, step)
        out_shape = []
        source_axis = 0
        for entry in expanded:
            if entry is None:
                out_shape.append(1)
                continue
            dim = shape[source_axis]
            if isinstance(entry, (bool, np.bool_)):
                raise IndexError(
                    "Boolean scalar indexing is not supported by lazy arrays."
                )
            if isinstance(entry, int):
                index = int(entry)
                if index < 0:
                    index += dim
                if not 0 <= index < dim:
                    raise IndexError(
                        "Index {} is out of bounds for axis {} with size {}.".format(
                            entry, source_axis, dim
                        )
                    )
                axis_specs.append(("int", source_axis, index))
            elif isinstance(entry, slice):
                start, stop, step = entry.indices(dim)
                if step == 0:
                    raise ValueError("slice step cannot be zero")
                axis_specs.append(("slice", source_axis, start, stop, step))
                out_shape.append(len(range(start, stop, step)))
            else:
                raise IndexError(
                    "Only integers, slices, None and Ellipsis are supported "
                    "by lazy array indexing, got {!r}.".format(entry)
                )
            source_axis += 1

        out_shape = tuple(out_shape)
        out_count = 1
        for dim in out_shape:
            out_count *= int(dim)

        if out_count == 0:
            # Empty selection: no chunk must be read.
            return self._finalize(np.empty(out_shape, dtype=header.dtype))

        # Bounding box of the selection in flat (storage order) coordinates,
        # used to prune chunks that cannot intersect the selection.
        lower = [0] * ndim
        upper = [0] * ndim
        for spec in axis_specs:
            if spec[0] == "int":
                _, axis, value = spec
                lower[axis] = upper[axis] = value
            else:
                _, axis, start, stop, step = spec
                selected = range(start, stop, step)
                first = selected[0]
                last = selected[-1]
                lower[axis] = min(first, last)
                upper[axis] = max(first, last)
        flat_min = int(np.ravel_multi_index(tuple(lower), shape, order=header.order))
        flat_max = int(np.ravel_multi_index(tuple(upper), shape, order=header.order))
        chunk_start = flat_min // header.chunk_elements
        chunk_end = min(header.n_chunks, flat_max // header.chunk_elements + 1)

        output = np.empty(out_count, dtype=header.dtype)

        for chunk_id in range(chunk_start, chunk_end):
            c0 = chunk_id * header.chunk_elements
            c1 = min(header.n_elements, c0 + header.chunk_elements)
            flat_positions = np.arange(c0, c1, dtype=np.int64)
            coords = np.unravel_index(flat_positions, shape, order=header.order)

            mask = np.ones(c1 - c0, dtype=bool)
            output_coords = []
            spec_index = 0
            for entry in expanded:
                if entry is None:
                    output_coords.append(0)
                    continue
                spec = axis_specs[spec_index]
                spec_index += 1
                if spec[0] == "int":
                    _, axis, value = spec
                    mask &= coords[axis] == value
                else:
                    _, axis, start, stop, step = spec
                    axis_coords = coords[axis]
                    if step > 0:
                        in_range = (axis_coords >= start) & (axis_coords < stop)
                    else:
                        in_range = (axis_coords <= start) & (axis_coords > stop)
                    divisible = (axis_coords - start) % step == 0
                    mask &= in_range & divisible
                    output_coords.append((axis_coords - start) // step)

            if not mask.any():
                # This chunk does not actually intersect the selection: do
                # not read it.
                continue

            chunk = self._source.read_chunk(self._base, header, chunk_id)
            if len(chunk) != c1 - c0:
                raise LazyArrayError("Corrupt chunk length while indexing.")
            # Coordinates are only meaningful for selected elements: clip
            # the computation to the mask to avoid out-of-bounds values
            # coming from non-selected chunk elements.
            selected_coords = [
                coords_axis[mask]
                if isinstance(coords_axis, np.ndarray)
                else coords_axis
                for coords_axis in output_coords
            ]
            output_flat = np.ravel_multi_index(
                np.broadcast_arrays(*selected_coords), out_shape
            )
            selected_values = chunk[mask]
            if out_shape == ():
                # Pure integer indexing: a single scalar element is selected.
                output[()] = selected_values[0]
            else:
                output[output_flat] = selected_values

        result = output.reshape(out_shape)
        result = self._finalize(result)
        if out_shape == ():
            # Pure integer indexing returns a numpy scalar, like ndarray does.
            return result[()]
        return result


def _rebuild_lazy_proxy(
    filename,
    identity,
    base,
    header_snapshot,
    cache_bytes,
    max_chunk_bytes,
    ensure_native,
):
    """Reconstruct a lazy proxy after pickle, reopening and verifying the file."""
    source = _source_for_filename(
        filename, cache_bytes=cache_bytes, max_chunk_bytes=max_chunk_bytes
    )
    try:
        if source.identity != identity:
            raise LazyArrayError(
                "Lazy array source file {!r} identity changed since the proxy "
                "was pickled; refusing to reopen it.".format(filename)
            )
        fh = source._file_handle
        with source._lock:
            fh.seek(base)
            header = parse_chunked_header(
                fh, base=base, max_chunk_bytes=source.max_chunk_bytes
            )
        if header.snapshot() != header_snapshot:
            raise LazyArrayError(
                "Chunked array header at offset {} of {!r} does not match the "
                "pickled proxy metadata.".format(base, filename)
            )
        proxy = LazyArrayProxy(
            source,
            base,
            header,
            ensure_native_byte_order=ensure_native,
        )
    except BaseException:
        source.release()
        raise
    return proxy


###############################################################################
# Persistable wrapper embedded in the pickle stream
###############################################################################


class ChunkedNumpyArrayWrapper(object):
    """Pickled placeholder carrying the chunked payload metadata."""

    def __init__(
        self,
        subclass,
        shape,
        order,
        dtype,
        chunk_elements,
        compressor_id,
        compress_level,
    ):
        self.subclass = subclass
        self.shape = tuple(shape)
        self.order = order
        self.dtype = dtype
        self.chunk_elements = int(chunk_elements)
        self.compressor_id = compressor_id
        self.compress_level = compress_level

    def _compress(self, raw):
        if self.compressor_id == _COMPRESSOR_RAW:
            return raw
        level = self.compress_level
        if level is None:
            return zlib.compress(raw)
        return zlib.compress(raw, level)

    def write_array(self, array, pickler):
        fh = pickler.file_handle
        if not (hasattr(fh, "seek") and hasattr(fh, "tell")):
            raise LazyArrayError(
                "The chunked array format requires a seekable writable file "
                "object or a filename."
            )
        if array.dtype.hasobject:
            raise LazyArrayError(
                "Arrays with object dtype cannot be written in the chunked "
                "format: use lazy_arrays=False to keep the legacy eager "
                "serialization path."
            )

        order = self.order
        flat = pickler.np.reshape(array, -1, order=order)
        n_elements = array.size
        n_chunks = (
            0
            if n_elements == 0
            else (n_elements + self.chunk_elements - 1) // self.chunk_elements
        )

        base = fh.tell()
        header_bytes, index_offset = _encode_header(
            self.compressor_id,
            order,
            self.shape,
            self.dtype,
            self.chunk_elements,
            [(0, 0, 0)] * n_chunks,
        )
        fh.write(_CHUNKED_MAGIC)
        fh.write(struct.pack("<I", len(header_bytes)))
        fh.write(header_bytes)
        data_start_rel = 8 + len(header_bytes)

        index = []
        cursor = data_start_rel
        for chunk_id in range(n_chunks):
            start = chunk_id * self.chunk_elements
            end = min(n_elements, start + self.chunk_elements)
            raw = flat[start:end].tobytes("C")
            encoded = self._compress(raw)
            index.append((cursor, len(encoded), len(raw)))
            fh.write(encoded)
            cursor += len(encoded)

        total_bytes = n_elements * self.dtype.itemsize
        fh.write(struct.pack("<Q", total_bytes))
        end_position = fh.tell()

        # Patch the placeholder index now that chunk sizes are known.
        fh.seek(base + 8 + index_offset)
        fh.write(_encode_index(index))
        fh.seek(end_position)

    def read(self, unpickler, ensure_native_byte_order):
        fh = unpickler.file_handle
        base = fh.tell()
        if unpickler.lazy_arrays:
            source = unpickler.get_lazy_source()
            header = parse_chunked_header(
                fh, base=base, max_chunk_bytes=source.max_chunk_bytes
            )
            payload_end = base + header.footer_rel + 8
            proxy = LazyArrayProxy(
                source,
                base,
                header,
                ensure_native_byte_order=ensure_native_byte_order,
                subclass=self.subclass,
            )
            fh.seek(payload_end)
            return proxy

        header = parse_chunked_header(fh, base=base)
        if header.n_elements == 0:
            array = unpickler.np.empty(0, dtype=header.dtype)
        else:
            array = unpickler.np.empty(header.n_elements, dtype=header.dtype)
            for chunk_id in range(header.n_chunks):
                start = chunk_id * header.chunk_elements
                end = min(header.n_elements, start + header.chunk_elements)
                offset, encoded_len, decoded_len = header.index[chunk_id]
                fh.seek(base + offset)
                encoded = _read_exact(
                    fh, encoded_len, "chunk {} encoded data".format(chunk_id)
                )
                if header.compressor_id == _COMPRESSOR_RAW:
                    decoded = encoded
                else:
                    try:
                        decoded = zlib.decompress(encoded)
                    except zlib.error as exc:
                        raise LazyArrayError(
                            "Failed to decompress chunk {}: {}".format(chunk_id, exc)
                        ) from exc
                if len(decoded) != decoded_len:
                    raise LazyArrayError(
                        "Chunk {} decoded length mismatch.".format(chunk_id)
                    )
                chunk = unpickler.np.frombuffer(
                    decoded,
                    dtype=header.dtype,
                    count=header.chunk_element_count(chunk_id),
                )
                array[start:end] = chunk
        if header.order == "F":
            array = array.reshape(header.shape[::-1]).transpose()
        else:
            array = array.reshape(header.shape)
        if ensure_native_byte_order:
            array = _ensure_native_byte_order(array)

        if hasattr(array, "__array_prepare__") and self.subclass not in (
            unpickler.np.ndarray,
            unpickler.np.memmap,
        ):
            new_array = _reconstruct(self.subclass, (0,), "b")
            array = new_array.__array_prepare__(array)

        fh.seek(base + header.footer_rel + 8)
        return array


###############################################################################
# Container walking helper
###############################################################################


def close_lazy_resources(obj, _seen=None):
    """Close every :class:`LazyArrayProxy` reachable from a container."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, LazyArrayProxy):
        obj.close()
        return
    obj_id = id(obj)
    if obj_id in _seen:
        return
    _seen.add(obj_id)
    if isinstance(obj, dict):
        for value in obj.values():
            close_lazy_resources(value, _seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            close_lazy_resources(value, _seen)
