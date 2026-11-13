"""
Tests for the filesystem-only layered Memory store backend.

A layered store combines one writable primary layer with an ordered list of
read-only fallback layers. These tests cover:

- complete-candidate resolution (function code + output + metadata in one
  single layer, never assembled across layers),
- function code identity and validation callback skipping,
- read-only fallback directories that are never modified,
- atomic promotion of fallback candidates to the primary layer,
- mmap binding to the selected layer's real file,
- thread safety of concurrent cross-layer resolutions,
- clear/reduce_size being scoped to the primary layer,
- invalid (duplicate/overlapping/nested) layer configurations.
"""

import os
import os.path
import pickle
import shutil
import threading

import pytest

from joblib import Memory
from joblib._store_backends import (
    CacheWarning,
    LayeredFileSystemStoreBackend,
)
from joblib.memory import _FUNCTION_HASHES
from joblib.test.common import np, with_numpy

# Module-level functions: their source code can be introspected reliably
# from every test (and from child threads/processes).
LM_CALLS = []


def lm_func(x):
    """Test function recording every actual execution."""
    LM_CALLS.append(x)
    return ("computed", x)


def lm_numpy(x):
    """Test function returning a numpy array for mmap tests."""
    return np.arange(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seed_cache(root, func, args, mmap_mode=None):
    """Build a regular single-layer cache and return its store directory.

    The in-process function hash cache is reset first so that the seeding
    Memory goes through the disk path and writes a self-contained
    ``func_code.py``, exactly like a cache produced in another process.
    """
    _FUNCTION_HASHES.clear()
    memory = Memory(location=str(root), verbose=0, mmap_mode=mmap_mode)
    cached = memory.cache(func)
    for arg in args:
        cached(arg)
    return os.path.join(str(root), "joblib")


def make_layered_memory(tmp_path, fallback=None, promote=False, primary=None,
                        mmap_mode=None, compress=False):
    if primary is None:
        primary = str(tmp_path / "primary")
    backend_options = {"promote": promote}
    if fallback is not None:
        backend_options["fallback"] = fallback
    return Memory(
        location=primary,
        backend="layered",
        verbose=0,
        mmap_mode=mmap_mode,
        compress=compress,
        backend_options=backend_options,
    )


def snapshot(root):
    """Return a {relative_path: content} snapshot of a directory tree."""
    content = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            with open(full, "rb") as f:
                content[os.path.relpath(full, root)] = f.read()
    return content


def call_id_of(cached, *args, **kwargs):
    return cached.func_id, cached._get_args_id(*args, **kwargs)


def item_dir_of(layer, call_id):
    return os.path.join(layer.location, *layer._split_id(call_id))


def find_item_dirs(root):
    item_dirs = []
    for dirpath, _, filenames in os.walk(root):
        if "output.pkl" in filenames:
            item_dirs.append(dirpath)
    return item_dirs


def write_primary_partial_item(primary_store, cached, arg, keep):
    """Locate the primary item directory for ``arg`` and remove all its
    files except those listed in ``keep``, producing an incomplete item."""
    call_id = call_id_of(cached, arg)
    backend = cached.store_backend
    item_dir = os.path.join(
        backend.location, *backend.primary._split_id(call_id)
    )
    for filename in os.listdir(item_dir):
        if filename not in keep:
            os.remove(os.path.join(item_dir, filename))
    return item_dir


# ---------------------------------------------------------------------------
# Basic resolution
# ---------------------------------------------------------------------------
def test_layered_fallback_hit_does_not_compute(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])

    LM_CALLS[:] = []
    cached = memory.cache(lm_func)
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []  # served from the fallback, no computation

    # With promotion disabled, a fallback hit writes nothing to the primary.
    assert find_item_dirs(memory.store_backend.primary.location) == []


def test_layered_primary_takes_precedence_with_distinct_sentinels(tmp_path):
    # Same call id in both layers, but a tampered fallback output: the
    # primary layer must always win and the fallback file is never read.
    primary_root = tmp_path / "prim_root"
    primary_store = seed_cache(primary_root, lm_func, [42])

    fallback_root = tmp_path / "fb_root"
    fallback_store = seed_cache(fallback_root, lm_func, [42])
    (fb_item,) = find_item_dirs(fallback_store)
    with open(os.path.join(fb_item, "output.pkl"), "wb") as f:
        f.write(b"garbage-primary-must-win-anyway")

    memory = make_layered_memory(
        tmp_path, primary=str(primary_root), fallback=[fallback_store]
    )
    assert memory.store_backend.primary.location == primary_store

    LM_CALLS[:] = []
    cached = memory.cache(lm_func)
    assert cached(42) == ("computed", 42)  # primary value, not the garbage
    assert LM_CALLS == []


def test_layered_no_cross_layer_assembly(tmp_path):
    # An incomplete primary item (output only, then metadata only) must not
    # be combined with pieces of the fallback layer: the complete fallback
    # candidate is used as a whole.
    primary_root = tmp_path / "prim_root"
    primary_store = seed_cache(primary_root, lm_func, [42])
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])

    memory = make_layered_memory(
        tmp_path, primary=str(primary_root), fallback=[fallback_store]
    )
    cached = memory.cache(lm_func)

    # Primary holds output.pkl but no metadata.json -> incomplete.
    write_primary_partial_item(
        primary_store, cached, 42, keep=["output.pkl"]
    )
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)  # whole fallback candidate
    assert LM_CALLS == []

    # Rebuild a complete primary, then keep metadata.json only.
    seed_cache(primary_root, lm_func, [42])
    write_primary_partial_item(
        primary_store, cached, 42, keep=["metadata.json"]
    )
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []


def test_layered_invalid_metadata_skips_candidate(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    (fb_item,) = find_item_dirs(fallback_store)
    with open(os.path.join(fb_item, "metadata.json"), "w") as f:
        f.write("this is not json")

    memory = make_layered_memory(tmp_path, fallback=[fallback_store])
    LM_CALLS[:] = []
    cached = memory.cache(lm_func)
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == [42]  # fallback skipped, function recomputed

    # The recomputation only populated the primary layer.
    assert len(find_item_dirs(memory.store_backend.primary.location)) == 1
    with open(os.path.join(fb_item, "metadata.json")) as f:
        assert f.read() == "this is not json"  # fallback untouched


def test_layered_ordered_fallbacks(tmp_path):
    fb1 = seed_cache(tmp_path / "fb1", lm_func, [42])
    fb2 = seed_cache(tmp_path / "fb2", lm_func, [42])

    # First fallback is incomplete: only its output.pkl remains.
    (fb1_item,) = find_item_dirs(fb1)
    os.remove(os.path.join(fb1_item, "metadata.json"))

    memory = make_layered_memory(tmp_path, fallback=[fb1, fb2])
    LM_CALLS[:] = []
    cached = memory.cache(lm_func)
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []
    resolved_layer, resolved_dir = memory.store_backend._resolve_candidate(
        call_id_of(cached, 42)
    )
    assert resolved_layer.location == fb2  # second fallback served

    # Make the first fallback complete again: it now shadows the second.
    shutil.copytree(fb2, fb1, dirs_exist_ok=True)
    resolved_layer, _ = memory.store_backend._resolve_candidate(
        call_id_of(cached, 42)
    )
    assert resolved_layer.location == fb1


# ---------------------------------------------------------------------------
# Function code identity
# ---------------------------------------------------------------------------
def test_layered_different_func_code_skips_and_preserves_fallback(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])

    # Tamper with the stored function code in the fallback layer, then take
    # the reference snapshot (the tampered tree is what must be preserved).
    func_id = Memory(location=str(tmp_path / "probe"), verbose=0).cache(
        lm_func
    ).func_id
    func_code_path = os.path.join(fallback_store, func_id, "func_code.py")
    with open(func_code_path, "w") as f:
        f.write("# first line: 1\ndef lm_func(x):\n    return 'stale code'\n")
    before = snapshot(fallback_store)

    # Force disk-based code validation (bypass the in-process hash cache).
    _FUNCTION_HASHES.clear()
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])
    LM_CALLS[:] = []
    cached = memory.cache(lm_func)
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == [42]  # stale fallback rejected, recomputed

    # Fallback directory is byte-for-byte unchanged (except the code we
    # tampered with ourselves): no deletion, no repair, no metadata edit.
    after = snapshot(fallback_store)
    assert after == before
    # Primary got the fresh result.
    assert len(find_item_dirs(memory.store_backend.primary.location)) == 1

    # Subsequent calls hit the primary; the stale fallback stays shadowed.
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []


def test_layered_missing_func_code_is_not_a_candidate(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    func_id = Memory(location=str(tmp_path / "probe"), verbose=0).cache(
        lm_func
    ).func_id
    os.remove(os.path.join(fallback_store, func_id, "func_code.py"))
    before = snapshot(fallback_store)

    _FUNCTION_HASHES.clear()
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])
    LM_CALLS[:] = []
    cached = memory.cache(lm_func)
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == [42]
    assert snapshot(fallback_store) == before


def test_layered_cached_clear_does_not_touch_fallback(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    before = snapshot(fallback_store)
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])
    cached = memory.cache(lm_func)

    cached.clear(warn=False)
    memory.clear(warn=False)
    assert snapshot(fallback_store) == before

    # The fallback candidate is still readable after clearing the primary.
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []


# ---------------------------------------------------------------------------
# Validation callback
# ---------------------------------------------------------------------------
def test_layered_validation_callback_accept_and_skip(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    before = snapshot(fallback_store)

    # Callback accepting the candidate: fallback hit, no computation.
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])
    cached = memory.cache(
        lm_func, cache_validation_callback=lambda metadata: True
    )
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []

    # Callback rejecting the only candidate: skip it, recompute and write
    # the primary layer only. The fallback is never modified.
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])
    cached = memory.cache(
        lm_func, cache_validation_callback=lambda metadata: False
    )
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == [42]
    assert snapshot(fallback_store) == before
    assert len(find_item_dirs(memory.store_backend.primary.location)) == 1


def test_layered_validation_callback_skips_to_next_layer(tmp_path):
    # Primary and fallback both hold a complete candidate for the same call
    # but with distinguishable sentinels.
    primary_root = tmp_path / "prim_root"
    primary_store = seed_cache(primary_root, lm_func, [42])
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    fb_snapshot = snapshot(fallback_store)

    memory = make_layered_memory(
        tmp_path, primary=str(primary_root), fallback=[fallback_store]
    )

    decisions = iter([False, True])  # reject primary, accept fallback

    def callback(metadata):
        return next(decisions)

    cached = memory.cache(lm_func, cache_validation_callback=callback)
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []
    # The rejected primary candidate was deleted (writable layer)...
    assert find_item_dirs(primary_store) == []
    # ...while the accepted fallback candidate was preserved.
    assert snapshot(fallback_store) == fb_snapshot


# ---------------------------------------------------------------------------
# Read-only fallback guarantees
# ---------------------------------------------------------------------------
def test_layered_fallback_is_never_written(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    before = snapshot(fallback_store)
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])

    # Any accidental write attempt on a fallback layer must blow up.
    for layer in memory.store_backend.fallbacks:
        def guarded_open(filename, mode="r", *args, **kwargs):
            if "w" in mode or "a" in mode or "+" in mode:
                raise AssertionError(
                    "a fallback layer must never be opened for writing"
                )
            return open(filename, mode, *args, **kwargs)

        layer._open_item = guarded_open
        layer.create_location = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("no location creation in a fallback layer")
        )
        layer.clear_location = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("no deletion in a fallback layer")
        )

    cached = memory.cache(lm_func)
    # Fallback hit, cache miss + computation, clear and size reduction:
    # none of these must mutate the fallback tree.
    assert cached(42) == ("computed", 42)
    LM_CALLS[:] = []
    cached(7)
    assert LM_CALLS == [7]
    cached.clear(warn=False)
    memory.clear(warn=False)
    memory.reduce_size(items_limit=0)
    assert snapshot(fallback_store) == before


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX permission bits are not enforced the same way on Windows",
)
def test_layered_read_only_permission_directory(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])

    def chmod_tree(root, dir_mode, file_mode):
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            for filename in filenames:
                os.chmod(os.path.join(dirpath, filename), file_mode)
            os.chmod(dirpath, dir_mode)
        os.chmod(root, dir_mode)

    chmod_tree(fallback_store, 0o555, 0o444)
    try:
        memory = make_layered_memory(tmp_path, fallback=[fallback_store])
        LM_CALLS[:] = []
        cached = memory.cache(lm_func)
        # Reading the read-only fallback works without any write attempt.
        assert cached(42) == ("computed", 42)
        assert LM_CALLS == []
        # A miss still computes normally (read-only fallback does not forbid
        # computation) and writes the primary layer.
        cached(7)
        assert LM_CALLS == [7]
        assert len(find_item_dirs(memory.store_backend.primary.location)) == 1
    finally:
        chmod_tree(fallback_store, 0o755, 0o644)


def test_layered_missing_fallback_dir_warns_and_is_skipped(tmp_path):
    missing = str(tmp_path / "does_not_exist")
    with pytest.warns(UserWarning, match="does not exist"):
        memory = make_layered_memory(tmp_path, fallback=[missing])
    assert not os.path.exists(missing)  # never created
    assert memory.store_backend.fallbacks == ()

    LM_CALLS[:] = []
    cached = memory.cache(lm_func)
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == [42]


# ---------------------------------------------------------------------------
# Promotion to the primary layer
# ---------------------------------------------------------------------------
def test_layered_promotion_copies_complete_candidate(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    before = snapshot(fallback_store)
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], promote=True
    )
    cached = memory.cache(lm_func)
    call_id = call_id_of(cached, 42)

    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []

    primary = memory.store_backend.primary
    promoted_dir = item_dir_of(primary, call_id)
    # The promoted item is complete: output AND metadata published together,
    # plus function code identity at the function level.
    assert os.path.isfile(os.path.join(promoted_dir, "output.pkl"))
    assert os.path.isfile(os.path.join(promoted_dir, "metadata.json"))
    assert os.path.isfile(
        os.path.join(primary.location, cached.func_id, "func_code.py")
    )
    # No promotion temporary is left behind.
    assert not os.path.exists(
        os.path.join(
            primary.location,
            LayeredFileSystemStoreBackend._promotion_tmp_dirname,
        )
    )
    assert snapshot(fallback_store) == before

    # A fresh layered memory over the promoted primary (no fallback) hits.
    primary_root = os.path.dirname(primary.location)
    standalone = make_layered_memory(
        tmp_path, primary=primary_root, fallback=[]
    )
    assert standalone.store_backend.primary.location == primary.location
    standalone_cached = standalone.cache(lm_func)
    assert standalone_cached.check_call_in_cache(42)
    assert standalone_cached(42) == ("computed", 42)


def test_layered_promotion_failure_keeps_fallback_serving(tmp_path, monkeypatch):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    before = snapshot(fallback_store)
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], promote=True
    )
    backend = memory.store_backend
    cached = memory.cache(lm_func)
    call_id = call_id_of(cached, 42)

    # Disk failure at the atomic publication step.
    def failing_move(src, dst):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(backend, "_move_item", failing_move)
    with pytest.warns(CacheWarning, match="Unable to promote"):
        assert cached(42) == ("computed", 42)

    # No fake hit: the primary target directory was never published.
    assert not os.path.exists(item_dir_of(backend.primary, call_id))
    # The temporary copy was cleaned up.
    tmp_root = os.path.join(
        backend.primary.location,
        LayeredFileSystemStoreBackend._promotion_tmp_dirname,
    )
    assert not os.path.exists(tmp_root) or os.listdir(tmp_root) == []
    # The untouched fallback candidate keeps serving every call.
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == []
    assert snapshot(fallback_store) == before

    # Disk failure while copying: same safety properties.
    import joblib._store_backends as sb

    def failing_copytree(src, dst, *args, **kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(sb.shutil, "copytree", failing_copytree)
    with pytest.warns(CacheWarning, match="Unable to promote"):
        assert cached(42) == ("computed", 42)
    assert not os.path.exists(item_dir_of(backend.primary, call_id))
    assert snapshot(fallback_store) == before


def test_layered_promotion_is_atomic_directory_publish(tmp_path, monkeypatch):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], promote=True
    )
    backend = memory.store_backend
    cached = memory.cache(lm_func)
    call_id = call_id_of(cached, 42)
    target = item_dir_of(backend.primary, call_id)
    observations = []

    real_move = backend._move_item

    def observing_move(src, dst):
        # At publication time the temp directory already holds the complete
        # candidate and the target does not exist yet: no half candidate can
        # ever be observed at the target path.
        observations.append(
            (
                os.path.isfile(os.path.join(src, "output.pkl")),
                os.path.isfile(os.path.join(src, "metadata.json")),
                os.path.exists(dst),
            )
        )
        assert dst == target
        return real_move(src, dst)

    monkeypatch.setattr(backend, "_move_item", observing_move)
    assert cached(42) == ("computed", 42)
    assert observations == [(True, True, False)]


def test_layered_concurrent_promotion(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    before = snapshot(fallback_store)
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], promote=True
    )
    cached = memory.cache(lm_func)
    backend = memory.store_backend
    call_id = call_id_of(cached, 42)
    barrier = threading.Barrier(8)
    results = []
    errors = []
    LM_CALLS[:] = []

    def worker():
        try:
            barrier.wait()
            for _ in range(5):
                results.append(cached(42))
        except Exception as e:  # pragma: no cover - failure reporting
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert results == [("computed", 42)] * 40
    assert LM_CALLS == []  # every thread was served from cache
    # Exactly one complete candidate in the primary layer.
    primary_items = find_item_dirs(backend.primary.location)
    assert len(primary_items) == 1
    assert os.path.isfile(os.path.join(primary_items[0], "output.pkl"))
    assert os.path.isfile(os.path.join(primary_items[0], "metadata.json"))
    assert item_dir_of(backend.primary, call_id) == primary_items[0]
    assert snapshot(fallback_store) == before


def test_layered_concurrent_resolutions_of_different_layers(tmp_path):
    # Call A lives in the primary layer, call B in the fallback layer.
    # Without any shared "current layer" variable, concurrent resolutions
    # must never swap files between the two calls.
    primary_root = tmp_path / "prim_root"
    seed_cache(primary_root, lm_func, [1])
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [2])

    memory = make_layered_memory(
        tmp_path, primary=str(primary_root), fallback=[fallback_store]
    )
    cached = memory.cache(lm_func)
    backend = memory.store_backend
    call_a = call_id_of(cached, 1)
    call_b = call_id_of(cached, 2)

    # Sanity check of the static layout.
    layer_a, dir_a = backend._resolve_candidate(call_a)
    layer_b, dir_b = backend._resolve_candidate(call_b)
    assert layer_a is backend.primary
    assert layer_b is backend.fallbacks[0]
    assert dir_a != dir_b

    LM_CALLS[:] = []
    stop = threading.Event()
    errors = []

    def worker_a():
        try:
            while not stop.is_set():
                assert cached(1) == ("computed", 1)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def worker_b():
        try:
            while not stop.is_set():
                assert cached(2) == ("computed", 2)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker_a), threading.Thread(target=worker_b)]
    for t in threads:
        t.start()
    stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join()
    assert errors == []
    assert LM_CALLS == []  # everything came from cache


# ---------------------------------------------------------------------------
# mmap binding
# ---------------------------------------------------------------------------
@with_numpy
def test_layered_mmap_binds_real_file_of_selected_layer(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_numpy, [6])
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], mmap_mode="r"
    )
    cached = memory.cache(lm_numpy)

    loaded = cached(6)
    assert isinstance(loaded, np.memmap)
    assert loaded.mode == "r"
    # The memmap is backed by the real fallback file, not a primary path.
    assert os.path.normcase(loaded.filename).startswith(
        os.path.normcase(fallback_store) + os.sep
    )
    np.testing.assert_array_equal(loaded, np.arange(6, dtype=np.float64))


@with_numpy
def test_layered_mmap_rebinds_after_promotion(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_numpy, [6])
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], promote=True, mmap_mode="r"
    )
    cached = memory.cache(lm_numpy)
    backend = memory.store_backend

    first = cached(6)
    assert os.path.normcase(first.filename).startswith(
        os.path.normcase(fallback_store) + os.sep
    )

    # Promotion copied the candidate; a new resolution now maps the primary
    # file (a different generation/location).
    second = cached(6)
    assert os.path.normcase(second.filename).startswith(
        os.path.normcase(backend.primary.location) + os.sep
    )
    np.testing.assert_array_equal(second, first)


@with_numpy
def test_layered_mmap_concurrent_different_layers(tmp_path):
    primary_root = tmp_path / "prim_root"
    primary_store = seed_cache(primary_root, lm_numpy, [3])
    fallback_store = seed_cache(tmp_path / "fb", lm_numpy, [5])
    memory = make_layered_memory(
        tmp_path,
        primary=str(primary_root),
        fallback=[fallback_store],
        mmap_mode="r",
    )
    cached = memory.cache(lm_numpy)
    errors = []

    def check(arg, expected_root):
        try:
            for _ in range(50):
                mapped = cached(arg)
                assert isinstance(mapped, np.memmap)
                assert os.path.normcase(mapped.filename).startswith(
                    os.path.normcase(expected_root) + os.sep
                )
                np.testing.assert_array_equal(
                    mapped, np.arange(arg, dtype=np.float64)
                )
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [
        threading.Thread(target=check, args=(3, primary_store)),
        threading.Thread(target=check, args=(5, fallback_store)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# ---------------------------------------------------------------------------
# clear / reduce_size scope
# ---------------------------------------------------------------------------
def test_layered_clear_only_affects_primary(tmp_path):
    primary_root = tmp_path / "prim_root"
    primary_store = seed_cache(primary_root, lm_func, [1, 2])
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [3])
    fb_before = snapshot(fallback_store)

    memory = make_layered_memory(
        tmp_path, primary=str(primary_root), fallback=[fallback_store]
    )

    assert len(find_item_dirs(primary_store)) == 2
    memory.clear(warn=False)
    assert find_item_dirs(primary_store) == []
    assert snapshot(fallback_store) == fb_before

    # The fallback layer still produces hits after the primary clear.
    cached = memory.cache(lm_func)
    LM_CALLS[:] = []
    assert cached(3) == ("computed", 3)
    assert LM_CALLS == []


def test_layered_reduce_size_only_affects_primary(tmp_path):
    primary_root = tmp_path / "prim_root"
    primary_store = seed_cache(primary_root, lm_func, [1, 2, 3])
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [4])
    fb_before = snapshot(fallback_store)

    memory = make_layered_memory(
        tmp_path, primary=str(primary_root), fallback=[fallback_store]
    )

    memory.reduce_size(items_limit=1)
    assert len(find_item_dirs(primary_store)) == 1
    assert snapshot(fallback_store) == fb_before

    memory.reduce_size(items_limit=0)
    assert find_item_dirs(primary_store) == []
    cached = memory.cache(lm_func)
    LM_CALLS[:] = []
    assert cached(4) == ("computed", 4)  # fallback still hit
    assert LM_CALLS == []


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------
def test_layered_rejects_duplicate_locations(tmp_path):
    fb = str(tmp_path / "fb")
    os.makedirs(fb)
    with pytest.raises(ValueError, match="twice"):
        make_layered_memory(tmp_path, fallback=[fb, fb])
    # Spelling differences resolving to the same directory are caught too.
    alias = os.path.join(fb, ".", "sub", "..")
    with pytest.raises(ValueError, match="twice|overlapping"):
        make_layered_memory(tmp_path, fallback=[fb, alias])


def test_layered_rejects_nested_locations(tmp_path):
    # Memory appends a "joblib" component to string locations: the actual
    # primary store is "<primary>/joblib". A fallback nested inside it must
    # be rejected.
    primary = str(tmp_path / "primary")
    nested_in_primary = os.path.join(primary, "joblib", "nested")
    with pytest.raises(ValueError, match="overlapping|nested"):
        make_layered_memory(tmp_path, primary=primary,
                            fallback=[nested_in_primary])

    # The other direction: a primary nested inside a fallback directory.
    fb = str(tmp_path / "fb")
    primary_inside_fb = os.path.join(fb, "child_root")
    with pytest.raises(ValueError, match="overlapping|nested"):
        Memory(
            location=primary_inside_fb,
            backend="layered",
            verbose=0,
            backend_options={"fallback": [fb]},
        )


def test_layered_rejects_symlink_aliases(tmp_path):
    fb_real = tmp_path / "fb_real"
    fb_real.mkdir()
    link = tmp_path / "fb_link"
    try:
        os.symlink(fb_real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not supported on this platform")
    with pytest.raises(ValueError, match="twice|overlapping"):
        make_layered_memory(tmp_path, fallback=[str(fb_real), str(link)])


# ---------------------------------------------------------------------------
# MemorizedResult and pickling
# ---------------------------------------------------------------------------
def test_layered_memorized_result_get_and_clear(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    before = snapshot(fallback_store)
    memory = make_layered_memory(tmp_path, fallback=[fallback_store])
    cached = memory.cache(lm_func)

    result = cached.call_and_shelve(42)
    assert result.get() == ("computed", 42)
    result.clear()  # primary-only clear: fallback must survive
    assert snapshot(fallback_store) == before
    assert result.get() == ("computed", 42)

    # MemorizedResult.get obeys the same identity rules: corrupting the
    # fallback metadata invalidates the candidate for it as well.
    (fb_item,) = find_item_dirs(fallback_store)
    with open(os.path.join(fb_item, "metadata.json"), "w") as f:
        f.write("{broken")
    LM_CALLS[:] = []
    assert cached(42) == ("computed", 42)
    assert LM_CALLS == [42]


def test_layered_backend_pickle_roundtrip(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], promote=True
    )
    backend = memory.store_backend

    reloaded = pickle.loads(pickle.dumps(backend))
    assert isinstance(reloaded, LayeredFileSystemStoreBackend)
    assert reloaded.location == backend.location
    assert [layer.location for layer in reloaded.fallbacks] == [
        fallback_store
    ]
    assert reloaded.promote is True

    # The reconstructed backend still resolves the fallback candidate.
    LM_CALLS[:] = []
    reloaded_memory = Memory(
        location=str(tmp_path / "reloaded_primary"),
        backend="layered",
        verbose=0,
    )
    reloaded_memory.store_backend = reloaded
    func = reloaded_memory.cache(lm_func)
    assert func(42) == ("computed", 42)
    assert LM_CALLS == []


def test_layered_repr_lists_layers(tmp_path):
    fallback_store = seed_cache(tmp_path / "fb", lm_func, [42])
    memory = make_layered_memory(
        tmp_path, fallback=[fallback_store], promote=True
    )
    text = repr(memory.store_backend)
    assert "LayeredFileSystemStoreBackend" in text
    assert fallback_store in text
    assert "promote=True" in text
    # The backend location is the "<location>/joblib" store directory.
    assert memory.store_backend.location == os.path.join(
        memory.location, "joblib"
    )
