"""Tests for the explicit chunked persistence format and lazy array proxies."""

import io
import pickle
import struct

import pytest

import joblib
from joblib import numpy_pickle
from joblib._lazy_arrays import (
    _CHUNKED_MAGIC,
    _SOURCE_REGISTRY,
    LazyArrayError,
    LazyArrayProxy,
    close_lazy_resources,
    get_lazy_options,
    lazy_options,
    set_lazy_options,
)
from joblib.test.common import np, with_numpy
from joblib.testing import parametrize, raises

###############################################################################
# Helpers


class RecordingFile:
    """File wrapper recording every seek/read so chunk access can be audited."""

    def __init__(self, file_handle):
        self.file_handle = file_handle
        self.ops = []

    def read(self, size=-1):
        position = self.file_handle.tell()
        data = self.file_handle.read(size)
        self.ops.append(("read", position, len(data)))
        return data

    def readline(self, *args):
        position = self.file_handle.tell()
        data = self.file_handle.readline(*args)
        self.ops.append(("readline", position, len(data)))
        return data

    def seek(self, position, whence=0):
        result = self.file_handle.seek(position, whence)
        self.ops.append(("seek", position, whence))
        return result

    def tell(self):
        return self.file_handle.tell()

    def seekable(self):
        return True

    def close(self):
        return self.file_handle.close()

    @property
    def closed(self):
        return self.file_handle.closed

    @property
    def name(self):
        return self.file_handle.name

    def read_offsets(self):
        return {position for kind, position, _ in self.ops if kind == "read"}

    def seek_offsets(self):
        return {position for kind, position, _ in self.ops if kind == "seek"}

    def chunk_reads(self):
        return [op for op in self.ops if op[0] == "read"]


class NonSeekableFile:
    def __init__(self, file_handle):
        self.file_handle = file_handle

    def read(self, size=-1):
        return self.file_handle.read(size)

    def readline(self, *args):
        return self.file_handle.readline(*args)


@pytest.fixture(autouse=True)
def reset_lazy_sources():
    yield
    for source in list(_SOURCE_REGISTRY.values()):
        source.close()
    _SOURCE_REGISTRY.clear()


def _dump(tmpdir, array, filename="data.pkl", **kwargs):
    path = tmpdir.join(filename).strpath
    joblib.dump(array, path, lazy_arrays=True, **kwargs)
    return path


def _index_file_offset(path, chunk_index=0):
    """Return the absolute file offset of a chunk index entry."""
    data = open(path, "rb").read()
    base = data.find(_CHUNKED_MAGIC)
    assert base >= 0
    header_length = struct.unpack("<I", data[base + 4 : base + 8])[0]
    stream = data[base + 8 : base + 8 + header_length]
    ndim = struct.unpack("<I", stream[4:8])[0]
    params_offset = 12 + 8 * ndim
    _, dtype_length, n_chunks = struct.unpack(
        "<QII", stream[params_offset : params_offset + 16]
    )
    index_start = params_offset + 16 + dtype_length
    return base + 8 + index_start + 24 * chunk_index, base, n_chunks


def _read_header(path):
    from joblib._lazy_arrays import parse_chunked_header

    with open(path, "rb") as fh:
        position = fh.read().find(_CHUNKED_MAGIC)
    with open(path, "rb") as fh:
        fh.seek(position)
        return parse_chunked_header(fh, base=position), position


###############################################################################
# Format and round-trip tests


@with_numpy
@parametrize("compressor", ["zlib", "none"])
def test_chunked_round_trip_matches_eager(tmpdir, compressor):
    array = np.arange(200, dtype=np.int64).reshape(20, 10)
    path = _dump(
        tmpdir,
        array,
        chunk_size=15,
        chunk_compressor=compressor,
    )
    eager = joblib.load(path)
    assert np.array_equal(eager, array)
    proxy = joblib.load(path, lazy_arrays=True)
    assert isinstance(proxy, LazyArrayProxy)
    assert proxy.shape == array.shape
    assert proxy.dtype == array.dtype
    assert proxy.ndim == array.ndim
    assert proxy.size == array.size
    assert proxy.nbytes == array.nbytes
    assert len(proxy) == array.shape[0]
    assert "LazyArrayProxy" in repr(proxy)
    # Full materialization is explicit and equivalent to eager loading.
    assert np.array_equal(np.asarray(proxy), array)
    assert np.array_equal(np.array(proxy, dtype=np.float64), array.astype(np.float64))
    close_lazy_resources(proxy)


@with_numpy
def test_chunked_short_and_single_block(tmpdir):
    array = np.arange(4, dtype=np.int32)
    # chunk_size larger than the array: a single, short block.
    path = _dump(tmpdir, array, chunk_size=1024)
    header, _ = _read_header(path)
    assert header.n_chunks == 1
    assert header.chunk_element_count(0) == 4
    proxy = joblib.load(path, lazy_arrays=True)
    assert np.array_equal(np.asarray(proxy), array)
    close_lazy_resources(proxy)


@with_numpy
def test_zero_length_array_has_no_chunks_and_no_reads(tmpdir):
    array = np.empty((0, 5), dtype=np.int16)
    path = _dump(tmpdir, array, chunk_size=3)
    header, _ = _read_header(path)
    assert header.n_chunks == 0
    assert header.total_bytes == 0

    with open(path, "rb") as raw:
        recorder = RecordingFile(raw)
        proxy = joblib.load(recorder, lazy_arrays=True)
        recorder.ops.clear()
        empty = proxy[1:3]
        assert empty.shape == (0, 5)
        assert empty.dtype == array.dtype
        assert recorder.chunk_reads() == []
        full = np.asarray(proxy)
        assert full.shape == array.shape
        assert recorder.chunk_reads() == []
        close_lazy_resources(proxy)


@with_numpy
def test_scalar_array(tmpdir):
    array = np.array(42, dtype=np.int64)
    path = _dump(tmpdir, array)
    proxy = joblib.load(path, lazy_arrays=True)
    assert proxy.shape == ()
    assert proxy[()] == 42
    assert np.array_equal(np.asarray(proxy), array)
    close_lazy_resources(proxy)


@with_numpy
@parametrize("order", ["C", "F"])
def test_storage_order_is_restored(tmpdir, order):
    array = np.asarray(
        np.arange(60).reshape(6, 10),
        order=order,
        dtype=np.float64,
    )
    path = _dump(tmpdir, array, chunk_size=13)
    header, _ = _read_header(path)
    assert header.order == order
    eager = joblib.load(path)
    assert np.array_equal(eager, array)
    assert getattr(eager.flags, "{}_contiguous".format(order.lower()))
    proxy = joblib.load(path, lazy_arrays=True, ensure_native_byte_order=False)
    materialized = np.asarray(proxy)
    assert getattr(materialized.flags, "{}_contiguous".format(order.lower()))
    assert np.array_equal(materialized, array)
    assert np.array_equal(proxy[::-1, 2:9:2], array[::-1, 2:9:2])
    close_lazy_resources(proxy)


@with_numpy
def test_big_endian_dtype_preserved_and_native_coercion(tmpdir):
    array = np.arange(20, dtype=">f8")
    path = _dump(tmpdir, array, chunk_size=7)
    proxy = joblib.load(path, lazy_arrays=True, ensure_native_byte_order=False)
    assert proxy.dtype == np.dtype(">f8")
    materialized = np.asarray(proxy)
    assert materialized.dtype == np.dtype(">f8")
    assert np.array_equal(materialized.view(np.uint8), array.view(np.uint8))
    assert np.array_equal(proxy[3:15:2], array[3:15:2])
    close_lazy_resources(proxy)

    native_proxy = joblib.load(path, lazy_arrays=True)
    native = np.asarray(native_proxy)
    assert native.dtype == array.dtype.newbyteorder("=")
    assert np.array_equal(native, array.astype(array.dtype.newbyteorder("=")))
    close_lazy_resources(native_proxy)


@with_numpy
def test_structured_dtype_with_subarrays(tmpdir):
    dtype = np.dtype([("x", "i4", (2, 2)), ("y", "u1")])
    array = np.zeros(10, dtype=dtype)
    array["x"] = np.arange(40).reshape(10, 2, 2)
    array["y"] = np.arange(10)
    path = _dump(tmpdir, array, chunk_size=3)
    header, _ = _read_header(path)
    assert header.dtype == dtype
    eager = joblib.load(path)
    assert np.array_equal(eager, array)
    proxy = joblib.load(path, lazy_arrays=True)
    assert np.array_equal(np.asarray(proxy), array)
    assert np.array_equal(proxy[2:8][["x"]], array[2:8][["x"]])
    assert proxy[4]["y"] == array[4]["y"]
    close_lazy_resources(proxy)


@with_numpy
def test_nested_container_eager_item_equivalence(tmpdir):
    arrays = {
        "a": np.arange(50, dtype=np.int32),
        "b": np.linspace(0, 1, 30).reshape(5, 6),
        "nested": [np.ones((3, 3), dtype=np.float32)],
        "meta": (1, "two", 3.0),
    }
    path = tmpdir.join("nested.pkl").strpath
    joblib.dump(arrays, path, lazy_arrays=True, chunk_size=11)
    eager = joblib.load(path)
    lazy = joblib.load(path, lazy_arrays=True)
    for key in ("a", "b"):
        assert np.array_equal(np.asarray(lazy[key]), eager[key])
    assert np.array_equal(np.asarray(lazy["nested"][0]), eager["nested"][0])
    assert lazy["meta"] == eager["meta"]
    close_lazy_resources(lazy)


###############################################################################
# Indexing behavior


@with_numpy
@parametrize(
    "index",
    [
        3,
        -2,
        slice(1, 4),
        slice(None, None, -1),
        slice(None, None, -3),
        (slice(None, None, -1), slice(None, None, -2)),
        (Ellipsis, slice(1, 7, 2)),
        (slice(2, 8), None),
        (None, Ellipsis, None),
        (),
        (slice(0, 0),),
        (slice(5, 5), slice(None)),
        (1, slice(None, None, -1)),
        (-1, -2, -1),
    ],
)
def test_basic_indexing_matches_ndarray(tmpdir, index):
    array = np.arange(120, dtype=np.int64).reshape(6, 5, 4)
    path = _dump(tmpdir, array, chunk_size=17)
    proxy = joblib.load(path, lazy_arrays=True)
    expected = array[index]
    result = proxy[index]
    assert np.array_equal(result, expected)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    close_lazy_resources(proxy)


@with_numpy
def test_integer_index_out_of_bounds(tmpdir):
    array = np.arange(10, dtype=np.int32)
    path = _dump(tmpdir, array, chunk_size=3)
    proxy = joblib.load(path, lazy_arrays=True)
    with raises(IndexError):
        proxy[10]
    with raises(IndexError):
        proxy[-11]
    close_lazy_resources(proxy)


@with_numpy
def test_only_intersecting_chunks_are_read(tmpdir):
    array = np.arange(100, dtype=np.int64).reshape(10, 10)
    path = _dump(tmpdir, array, chunk_size=20)  # 5 chunks of 20 elements
    header, base = _read_header(path)
    chunk_ranges = [
        (base + offset, base + offset + encoded_len)
        for offset, encoded_len, _ in header.index
    ]

    with open(path, "rb") as raw:
        recorder = RecordingFile(raw)
        proxy = joblib.load(recorder, lazy_arrays=True)
        recorder.ops.clear()

        result = proxy[5:7]  # flat elements 50..79 -> chunks 2 and 3
        assert np.array_equal(result, array[5:7])

        touched_chunks = {
            chunk_id
            for chunk_id, (start, end) in enumerate(chunk_ranges)
            if any(start <= position < end for position in recorder.seek_offsets())
        }
        assert touched_chunks == {2, 3}

        # Empty slices must not trigger any chunk read.
        recorder.ops.clear()
        empty = proxy[0:0]
        assert empty.shape == (0, 10)
        assert recorder.chunk_reads() == []

        # A disjoint region does not read previously selected chunks either.
        recorder.ops.clear()
        _ = proxy[0:2]  # flat elements 0..19 -> chunk 0 only
        touched_chunks = {
            chunk_id
            for chunk_id, (start, end) in enumerate(chunk_ranges)
            if any(start <= position < end for position in recorder.seek_offsets())
        }
        assert touched_chunks == {0}
        close_lazy_resources(proxy)


@with_numpy
def test_cross_block_reverse_access(tmpdir):
    array = np.arange(100, dtype=np.int64).reshape(10, 10)
    path = _dump(tmpdir, array, chunk_size=20)
    header, base = _read_header(path)
    chunk_ranges = [
        (base + offset, base + offset + encoded_len)
        for offset, encoded_len, _ in header.index
    ]
    with open(path, "rb") as raw:
        recorder = RecordingFile(raw)
        proxy = joblib.load(recorder, lazy_arrays=True)
        recorder.ops.clear()
        reversed_array = proxy[::-1]
        assert np.array_equal(reversed_array, array[::-1])
        touched_chunks = {
            chunk_id
            for chunk_id, (start, end) in enumerate(chunk_ranges)
            if any(start <= position < end for position in recorder.seek_offsets())
        }
        # Every chunk intersects the fully reversed selection.
        assert touched_chunks == set(range(header.n_chunks))
        assert np.array_equal(proxy[::-2, ::-3], array[::-2, ::-3])
        close_lazy_resources(proxy)


@with_numpy
def test_getitem_does_not_materialize_everything(tmpdir):
    # A large-ish array with a tiny selection: only one chunk must be read.
    array = np.arange(2000, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=100)
    with open(path, "rb") as raw:
        recorder = RecordingFile(raw)
        proxy = joblib.load(recorder, lazy_arrays=True)
        recorder.ops.clear()
        assert proxy[5] == 5
        data_reads = recorder.chunk_reads()
        assert len(data_reads) == 1
        close_lazy_resources(proxy)


###############################################################################
# LRU cache and resource limits


@with_numpy
def test_lru_repeated_access_hits_cache(tmpdir):
    array = np.arange(100, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=20)
    with open(path, "rb") as raw:
        recorder = RecordingFile(raw)
        proxy = joblib.load(recorder, lazy_arrays=True)
        _ = proxy[40:65]  # warm chunks 2 (40..59) and 3 (60..79)
        recorder.ops.clear()
        for _ in range(5):
            assert np.array_equal(proxy[40:65], array[40:65])
        assert recorder.chunk_reads() == []
        info = proxy.source.cache_info()
        assert info["entries"] == 2
        assert info["retained_bytes"] == 2 * 20 * array.itemsize
        close_lazy_resources(proxy)


@with_numpy
def test_lru_byte_budget_is_enforced(tmpdir):
    array = np.arange(100, dtype=np.int64)  # 800 raw bytes, 10 chunks of 80
    path = _dump(tmpdir, array, chunk_size=10)
    proxy = joblib.load(path, lazy_arrays=True, cache_bytes=200)
    for start in range(0, 100, 10):
        _ = proxy[start : start + 10]
    info = proxy.source.cache_info()
    assert info["retained_bytes"] <= 200
    assert info["entries"] <= 2
    # Evicted chunks can be decoded again on demand.
    assert np.array_equal(proxy[0:10], array[0:10])
    assert proxy.source.cache_info()["retained_bytes"] <= 200
    close_lazy_resources(proxy)


@with_numpy
def test_lru_budget_counts_retained_bytes_not_entries(tmpdir):
    array = np.arange(100, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=10)
    proxy = joblib.load(path, lazy_arrays=True, cache_bytes=190)
    _ = proxy[0:10]
    info = proxy.source.cache_info()
    # A single 80-byte chunk fits, even though a second one would not.
    assert info["entries"] == 1
    assert info["retained_bytes"] == 80
    _ = proxy[10:20]
    info = proxy.source.cache_info()
    assert info["retained_bytes"] <= 190
    # A chunk larger than the budget (80 bytes each, budget 60) but below
    # max_chunk_bytes still decodes on demand.
    close_lazy_resources(proxy)
    proxy = joblib.load(path, lazy_arrays=True, cache_bytes=60)
    assert np.array_equal(proxy[0:10], array[0:10])
    info = proxy.source.cache_info()
    assert info["entries"] == 1
    assert info["retained_bytes"] == 80
    close_lazy_resources(proxy)


@with_numpy
def test_oversized_chunk_declaration_rejected_before_allocation(tmpdir):
    array = np.arange(100, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=10)
    # The rejection happens while parsing the header, before any getitem.
    with raises(LazyArrayError, match="max_chunk_bytes"):
        joblib.load(path, lazy_arrays=True, max_chunk_bytes=40)


@with_numpy
def test_lazy_options_configuration(tmpdir):
    previous = get_lazy_options()
    try:
        set_lazy_options(cache_bytes=1024, max_chunk_bytes=2048)
        assert get_lazy_options() == {
            "cache_bytes": 1024,
            "max_chunk_bytes": 2048,
        }
        array = np.arange(40, dtype=np.int64)
        path = _dump(tmpdir, array, chunk_size=10)
        proxy = joblib.load(path, lazy_arrays=True)
        assert proxy.source.cache_bytes == 1024
        assert proxy.source.max_chunk_bytes == 2048
        close_lazy_resources(proxy)
        with lazy_options(cache_bytes=512):
            proxy = joblib.load(path, lazy_arrays=True)
            assert proxy.source.cache_bytes == 512
            close_lazy_resources(proxy)
        assert get_lazy_options()["cache_bytes"] == 1024
        with raises(ValueError):
            set_lazy_options(cache_bytes=0)
        with raises(ValueError):
            set_lazy_options(max_chunk_bytes=-1)
    finally:
        set_lazy_options(
            cache_bytes=previous["cache_bytes"],
            max_chunk_bytes=previous["max_chunk_bytes"],
        )


###############################################################################
# Corruption and validation


@with_numpy
def test_corrupt_magic_is_rejected(tmpdir):
    array = np.arange(20, dtype=np.int32)
    path = _dump(tmpdir, array, chunk_size=5)
    data = bytearray(open(path, "rb").read())
    position = data.find(_CHUNKED_MAGIC)
    data[position : position + 4] = b"XXXX"
    corrupt_path = tmpdir.join("bad_magic.pkl").strpath
    open(corrupt_path, "wb").write(bytes(data))
    with raises(LazyArrayError):
        joblib.load(corrupt_path, lazy_arrays=True)


@with_numpy
def test_corrupt_index_encoded_length_is_rejected(tmpdir):
    array = np.arange(100, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=10)
    entry_offset, _, _ = _index_file_offset(path)
    data = bytearray(open(path, "rb").read())
    encoded_len = struct.unpack(
        "<Q", bytes(data[entry_offset + 8 : entry_offset + 16])
    )[0]
    data[entry_offset + 8 : entry_offset + 16] = struct.pack("<Q", encoded_len + 10**9)
    corrupt_path = tmpdir.join("bad_index.pkl").strpath
    open(corrupt_path, "wb").write(bytes(data))
    with raises(LazyArrayError):
        joblib.load(corrupt_path, lazy_arrays=True)


@with_numpy
def test_corrupt_index_decoded_length_is_rejected(tmpdir):
    array = np.arange(100, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=10)
    entry_offset, _, _ = _index_file_offset(path)
    data = bytearray(open(path, "rb").read())
    decoded_len = struct.unpack(
        "<Q", bytes(data[entry_offset + 16 : entry_offset + 24])
    )[0]
    data[entry_offset + 16 : entry_offset + 24] = struct.pack("<Q", decoded_len + 8)
    corrupt_path = tmpdir.join("bad_decoded.pkl").strpath
    open(corrupt_path, "wb").write(bytes(data))
    with raises(LazyArrayError):
        joblib.load(corrupt_path, lazy_arrays=True)


@with_numpy
def test_truncated_file_is_rejected(tmpdir):
    array = np.arange(100, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=10)
    data = open(path, "rb").read()
    trunc_path = tmpdir.join("trunc.pkl").strpath
    open(trunc_path, "wb").write(data[:-50])
    with raises(LazyArrayError):
        joblib.load(trunc_path, lazy_arrays=True)


@with_numpy
def test_corrupt_chunk_data_raises_without_zero_fill(tmpdir):
    array = np.arange(100, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=10)
    header, base = _read_header(path)
    first_offset, _, _ = header.index[0]
    data = bytearray(open(path, "rb").read())
    data_position = base + first_offset
    data[data_position] = data[data_position] ^ 0xFF
    corrupt_path = tmpdir.join("bad_chunk.pkl").strpath
    open(corrupt_path, "wb").write(bytes(data))
    proxy = joblib.load(corrupt_path, lazy_arrays=True)
    with raises(LazyArrayError):
        _ = proxy[0:10]
    close_lazy_resources(proxy)


###############################################################################
# Compatibility rules and explicit refusals


@with_numpy
def test_legacy_file_lazy_request_is_explicitly_rejected(tmpdir):
    array = np.arange(10, dtype=np.int32)
    path = tmpdir.join("legacy.pkl").strpath
    joblib.dump(array, path)  # legacy contiguous format
    with raises(LazyArrayError, match="chunked format"):
        joblib.load(path, lazy_arrays=True)
    # Eager loading still works.
    assert np.array_equal(joblib.load(path), array)


@with_numpy
def test_legacy_compressed_file_lazy_request_is_rejected(tmpdir):
    array = np.arange(10, dtype=np.int32)
    path = tmpdir.join("legacy.gz").strpath
    joblib.dump(array, path, compress=3)
    with raises(LazyArrayError):
        joblib.load(path, lazy_arrays=True)


@with_numpy
def test_object_dtype_rejected_at_dump(tmpdir):
    array = np.array(["alpha", "beta"], dtype=object)
    path = tmpdir.join("objects.pkl").strpath
    with raises(LazyArrayError, match="object dtype"):
        joblib.dump(array, path, lazy_arrays=True)
    # The legacy eager path still handles object arrays.
    joblib.dump(array, path)
    assert np.array_equal(joblib.load(path), array)


@with_numpy
def test_lazy_and_mmap_mode_are_mutually_exclusive(tmpdir):
    array = np.arange(10, dtype=np.int32)
    path = _dump(tmpdir, array)
    with raises(ValueError, match="mutually exclusive"):
        joblib.load(path, lazy_arrays=True, mmap_mode="r")


@with_numpy
def test_lazy_and_whole_file_compression_are_mutually_exclusive(tmpdir):
    path = tmpdir.join("compressed.pkl").strpath
    with raises(ValueError, match="incompatible"):
        joblib.dump(np.arange(5), path, compress=3, lazy_arrays=True)
    with raises(ValueError):
        joblib.dump(np.arange(5), path, compress=True, lazy_arrays=True)


@with_numpy
def test_non_seekable_file_object_rejected(tmpdir):
    array = np.arange(10, dtype=np.int32)
    path = _dump(tmpdir, array)
    with open(path, "rb") as raw:
        with raises(LazyArrayError, match="seekable"):
            joblib.load(NonSeekableFile(raw), lazy_arrays=True)


@with_numpy
def test_invalid_resource_limits_rejected(tmpdir):
    array = np.arange(10, dtype=np.int32)
    path = _dump(tmpdir, array)
    with raises(ValueError):
        joblib.load(path, lazy_arrays=True, cache_bytes=0)
    with raises(ValueError):
        joblib.load(path, lazy_arrays=True, max_chunk_bytes=-5)
    with raises(ValueError):
        joblib.load(path, lazy_arrays=True, cache_bytes=True)


@with_numpy
def test_invalid_chunk_size_rejected(tmpdir):
    path = tmpdir.join("bad_chunk.pkl").strpath
    with raises(ValueError):
        joblib.dump(np.arange(5), path, lazy_arrays=True, chunk_size=0)


###############################################################################
# File resource lifecycle


@with_numpy
def test_proxies_share_source_and_close_independently(tmpdir):
    arrays = {
        "a": np.arange(50, dtype=np.int32),
        "b": np.arange(100, 200, dtype=np.int32),
    }
    path = tmpdir.join("shared.pkl").strpath
    joblib.dump(arrays, path, lazy_arrays=True, chunk_size=15)
    loaded = joblib.load(path, lazy_arrays=True)
    proxy_a, proxy_b = loaded["a"], loaded["b"]
    assert proxy_a.source is proxy_b.source
    source = proxy_a.source
    assert source._refcount == 2
    proxy_a.close()
    # The other proxy must keep working and keep the source open.
    assert not source.closed
    assert np.array_equal(np.asarray(proxy_b), arrays["b"])
    proxy_b.close()
    assert source.closed
    with raises(LazyArrayError):
        _ = proxy_b[0]


@with_numpy
def test_close_lazy_resources_walks_containers(tmpdir):
    arrays = {
        "x": [np.arange(5, dtype=np.int16)],
        "y": (np.ones(4, dtype=np.int32), np.zeros(2, dtype=np.int32)),
    }
    path = tmpdir.join("containers.pkl").strpath
    joblib.dump(arrays, path, lazy_arrays=True, chunk_size=2)
    loaded = joblib.load(path, lazy_arrays=True)
    source = loaded["x"][0].source
    close_lazy_resources(loaded)
    assert source.closed


@with_numpy
def test_filename_handle_is_owned_and_closed(tmpdir):
    array = np.arange(10, dtype=np.int32)
    path = _dump(tmpdir, array, chunk_size=3)
    proxy = joblib.load(path, lazy_arrays=True)
    source = proxy.source
    assert source.owns_handle
    assert not source._file_handle.closed
    proxy.close()
    assert source._file_handle.closed


@with_numpy
def test_user_file_object_is_not_owned(tmpdir):
    array = np.arange(30, dtype=np.int32)
    buffer = io.BytesIO()
    joblib.dump(array, buffer, lazy_arrays=True, chunk_size=7)
    buffer.seek(0)
    proxy = joblib.load(buffer, lazy_arrays=True)
    assert not proxy.source.owns_handle
    assert np.array_equal(proxy[5:20], array[5:20])
    proxy.close()
    # User-provided objects remain usable after proxy close.
    assert not buffer.closed
    buffer.seek(0)
    assert len(buffer.read(4)) == 4


@with_numpy
def test_proxy_pickle_reopens_and_verifies_identity(tmpdir):
    array = np.arange(50, dtype=np.int32)
    path = _dump(tmpdir, array, chunk_size=10)
    proxy = joblib.load(path, lazy_arrays=True)
    reloaded = pickle.loads(pickle.dumps(proxy))
    assert isinstance(reloaded, LazyArrayProxy)
    assert np.array_equal(np.asarray(reloaded), array)
    assert np.array_equal(reloaded[3:30:2], array[3:30:2])
    close_lazy_resources(proxy)
    close_lazy_resources(reloaded)

    # Rewriting the file with different content invalidates pickled proxies.
    proxy = joblib.load(path, lazy_arrays=True)
    stale_blob = pickle.dumps(proxy)
    joblib.dump(np.arange(25, dtype=np.int32), path, lazy_arrays=True, chunk_size=5)
    with raises(LazyArrayError, match="identity|header"):
        pickle.loads(stale_blob)
    close_lazy_resources(proxy)


@with_numpy
def test_file_object_backed_proxy_cannot_be_pickled(tmpdir):
    array = np.arange(20, dtype=np.int32)
    buffer = io.BytesIO()
    joblib.dump(array, buffer, lazy_arrays=True, chunk_size=5)
    buffer.seek(0)
    proxy = joblib.load(buffer, lazy_arrays=True)
    with raises(TypeError, match="filename"):
        pickle.dumps(proxy)
    close_lazy_resources(proxy)


@with_numpy
def test_repeated_loads_share_registered_source(tmpdir):
    array = np.arange(30, dtype=np.int32)
    path = _dump(tmpdir, array, chunk_size=6)
    first = joblib.load(path, lazy_arrays=True)
    second = joblib.load(path, lazy_arrays=True)
    assert first.source is second.source
    _ = first[0:6]
    # The second proxy benefits from the first one's cached chunks.
    assert np.array_equal(np.asarray(second), array)
    close_lazy_resources(first)
    assert not second.source.closed
    close_lazy_resources(second)
    assert second.source.closed


@with_numpy
def test_payload_without_arrays_does_not_leak_handle(tmpdir):
    path = tmpdir.join("no_arrays.pkl").strpath
    joblib.dump({"a": 1, "b": [1, 2, 3]}, path, lazy_arrays=True)
    result = joblib.load(path, lazy_arrays=True)
    assert result == {"a": 1, "b": [1, 2, 3]}
    assert _SOURCE_REGISTRY == {}


@with_numpy
def test_pathlike_inputs(tmpdir):
    from pathlib import Path

    array = np.arange(12, dtype=np.int32)
    path = Path(tmpdir.join("pathlike.pkl").strpath)
    joblib.dump(array, path, lazy_arrays=True, chunk_size=4)
    proxy = joblib.load(path, lazy_arrays=True)
    assert np.array_equal(np.asarray(proxy), array)
    close_lazy_resources(proxy)
    assert _SOURCE_REGISTRY == {}


@with_numpy
def test_concurrent_proxy_access_is_thread_safe(tmpdir):
    import threading

    array = np.arange(200, dtype=np.int64)
    path = _dump(tmpdir, array, chunk_size=13)
    proxy = joblib.load(path, lazy_arrays=True, cache_bytes=500)
    errors = []

    def worker(worker_id):
        try:
            for start in range(worker_id, 100, 7):
                assert np.array_equal(
                    proxy[start : start + 10], array[start : start + 10]
                )
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    info = proxy.source.cache_info()
    assert info["retained_bytes"] <= info["byte_budget"]
    close_lazy_resources(proxy)


###############################################################################
# Public API surface


@with_numpy
def test_public_api_imports():
    assert hasattr(joblib, "LazyArrayProxy")
    assert hasattr(joblib, "close_lazy_resources")
    assert hasattr(joblib, "get_lazy_options")
    assert hasattr(joblib, "set_lazy_options")
    assert hasattr(joblib, "lazy_options")
    assert numpy_pickle.LazyArrayProxy is LazyArrayProxy
