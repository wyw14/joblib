"""Tests for the generation-layout store and cross-process result pinning.

The pinning feature is opt-in through ``Memory(..., backend='local-generation')``.
These tests verify protection with *real* files and result values (never the
mere presence of lease markers), including two-process barrier races,
recomputation into new generations, logical clears, owner termination, PID
reuse safety and multi-owner leases.
"""

import datetime
import json
import os
import pickle
import subprocess
import sys
import time

import pytest

from joblib import MemorizedResult, Memory, ResultLease, numpy_pickle
from joblib._pinning import (
    SESSIONS_DIRNAME,
    _acquire_session,
    _close_session,
    session_is_alive,
)
from joblib._store_backends import (
    GENERATION_BACKEND_NAME,
    GENERATIONS_DIRNAME,
    GenerationFileSystemStoreBackend,
)
from joblib.test.common import with_numpy

GENERATION_BACKEND = GENERATION_BACKEND_NAME


def make_memory(tmp_path, **kwargs):
    options = dict(verbose=0, backend=GENERATION_BACKEND)
    options.update(kwargs)
    return Memory(location=str(tmp_path), **options)


def output_pkl_path(backend, call_id, generation):
    item_path = backend._item_path(call_id)
    return os.path.join(backend._generation_path(item_path, generation), "output.pkl")


def list_generations(backend, call_id):
    item_path = backend._item_path(call_id)
    return sorted(os.listdir(os.path.join(item_path, GENERATIONS_DIRNAME)))


# ---------------------------------------------------------------------------
# In-process behavior
# ---------------------------------------------------------------------------
def test_pin_blocks_reduce_size_and_release_reclaims(tmp_path):
    memory = make_memory(tmp_path)
    calls = []

    @memory.cache
    def f(x):
        calls.append(x)
        return "value-{}".format(x)

    results = [f.call_and_shelve(i) for i in range(3)]
    pinned = results[0]
    lease = pinned.pin()
    assert isinstance(lease, ResultLease)
    pinned_path = output_pkl_path(
        memory.store_backend, pinned._call_id, pinned.generation
    )

    # Every logical item is an eviction candidate but one is protected.
    report = memory.reduce_size(items_limit=0)
    assert report["protected"] == [memory.store_backend._item_path(pinned._call_id)]
    assert report["unmet_items"] == 1
    assert report["pinned_bytes"] > 0
    assert os.path.exists(pinned_path), "pinned output.pkl was deleted"
    assert pinned.get() == "value-0"
    for result in results[1:]:
        with pytest.raises(KeyError):
            result.get()

    # Once released, ordinary eviction rules reclaim the generation.
    lease.release()
    report = memory.reduce_size(items_limit=0)
    assert report["protected"] == []
    assert not os.path.exists(pinned_path)
    with pytest.raises(KeyError):
        pinned.get()


def test_recomputation_creates_distinct_generations_and_fixes_old_result(tmp_path):
    memory = make_memory(tmp_path)
    counter = {"n": 0}

    @memory.cache
    def f(x):
        counter["n"] += 1
        return (x, counter["n"])

    first = f.call_and_shelve("a")
    first_gen = first.generation
    lease = first.pin()

    # Forced recomputation of the same call_id must not overwrite anything.
    f.call("a")
    second = f.call_and_shelve("a")
    second_gen = second.generation
    assert second_gen != first_gen

    generations = list_generations(memory.store_backend, first._call_id)
    assert generations == sorted([first_gen, second_gen])
    for generation, expected in ((first_gen, ("a", 1)), (second_gen, ("a", 2))):
        gen_dir = memory.store_backend._generation_path(
            memory.store_backend._item_path(first._call_id), generation
        )
        # Every generation is self-contained, including function identity.
        for filename in ("output.pkl", "metadata.json", "func_code.py"):
            assert os.path.exists(os.path.join(gen_dir, filename))
        with open(os.path.join(gen_dir, "output.pkl"), "rb") as fh:
            assert numpy_pickle.load(fh) == expected

    # The old MemorizedResult keeps reading the generation it was shelved on.
    assert first.get() == ("a", 1)
    assert second.get() == ("a", 2)
    assert f("a") == ("a", 2)

    # Unpinned old generation would be reaped; the pinned one stays.
    memory.reduce_size(items_limit=0)
    assert list_generations(memory.store_backend, first._call_id) == [first_gen]
    assert first.get() == ("a", 1)
    lease.release()


def test_func_clear_logically_invalidates_but_keeps_pinned_generation(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x + 1

    cached = memory.cache(f)
    result = cached.call_and_shelve(1)
    generation = result.generation
    lease = result.pin()
    pinned_path = output_pkl_path(memory.store_backend, result._call_id, generation)

    cached.clear()
    assert not cached.check_call_in_cache(1)
    assert os.path.exists(pinned_path)
    assert result.get() == 2  # still the fixed, physically retained value

    # A fresh computation publishes a new generation, the old one survives.
    new_result = cached.call_and_shelve(1)
    assert new_result.generation != generation
    assert new_result.get() == 2
    assert result.get() == 2
    assert generation in list_generations(memory.store_backend, result._call_id)
    lease.release()


def test_memory_clear_preserves_pinned_content_and_its_function_identity(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x * 3

    result = f.call_and_shelve(4)
    generation = result.generation
    lease = result.pin()
    gen_dir = memory.store_backend._generation_path(
        memory.store_backend._item_path(result._call_id), generation
    )

    memory.clear(warn=False)
    assert not memory.store_backend.contains_item(result._call_id)
    # Physical retention: real files and real values, not just markers.
    assert os.path.exists(os.path.join(gen_dir, "output.pkl"))
    assert os.path.exists(os.path.join(gen_dir, "func_code.py"))
    assert result.get() == 12

    # A whole new cache can be built on top while the old result stays fixed.
    @memory.cache
    def f(x):  # noqa: F811
        return x * 4

    new_result = f.call_and_shelve(4)
    assert new_result.get() == 16
    assert result.get() == 12
    lease.release()


def test_nested_leases_multi_owner_and_idempotent_release(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x

    result = f.call_and_shelve(1)
    lease_a = result.pin()
    lease_b = result.pin()
    assert lease_a.lease_id != lease_b.lease_id

    lease_a.release()
    lease_a.release()  # idempotent, must not touch the other owner
    report = memory.reduce_size(items_limit=0)
    assert report["protected"] == [memory.store_backend._item_path(result._call_id)]
    assert result.get() == 1

    lease_b.release()
    lease_b.release()  # idempotent again
    memory.reduce_size(items_limit=0)
    with pytest.raises(KeyError):
        result.get()


def test_pin_as_context_manager(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x

    result = f.call_and_shelve(99)
    with result.pin() as lease:
        assert lease.leased
        memory.reduce_size(items_limit=0)
        assert result.get() == 99
    assert not lease.leased
    memory.reduce_size(items_limit=0)
    with pytest.raises(KeyError):
        result.get()


def test_bytes_limit_reports_unmet_target_without_force_deleting(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return "x" * 2000

    result = f.call_and_shelve(1)
    lease = result.pin()
    report = memory.reduce_size(bytes_limit=10)
    assert report["bytes_limit"] == 10
    assert report["unmet_bytes"] > 0
    assert report["protected"]
    assert result.get() == "x" * 2000

    lease.release()
    report = memory.reduce_size(bytes_limit=10)
    assert report["protected"] == []
    assert report["unmet_bytes"] == 0
    with pytest.raises(KeyError):
        result.get()


def test_age_limit_reports_protected_old_items(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x

    result = f.call_and_shelve(1)
    lease = result.pin()
    time.sleep(0.01)
    report = memory.reduce_size(age_limit=datetime.timedelta(seconds=0))
    assert report["age_unmet"] == [memory.store_backend._item_path(result._call_id)]
    assert result.get() == 1
    lease.release()


def test_pin_on_missing_generation_fails_atomically(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x

    result = f.call_and_shelve(1)
    assert result.generation is not None
    memory.reduce_size(items_limit=0)

    # The result now points at a generation that no longer exists.
    stale = MemorizedResult(
        memory.store_backend.location,
        result._call_id,
        backend=GENERATION_BACKEND,
        generation=result.generation,
    )
    with pytest.raises(KeyError):
        stale.pin()
    # A failed acquisition leaves no marker behind.
    leases_dir = os.path.join(
        memory.store_backend._item_path(result._call_id), "leases"
    )
    assert not os.path.exists(leases_dir) or not os.listdir(leases_dir)


def test_repr_and_pickle_roundtrip_keep_the_generation(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x + 7

    result = f.call_and_shelve(1)
    restored = pickle.loads(pickle.dumps(result))
    assert restored.generation == result.generation
    assert restored.get() == 8

    via_repr = eval(repr(result), {"MemorizedResult": MemorizedResult})
    assert via_repr.generation == result.generation
    assert via_repr.get() == 8
    assert repr(via_repr) == repr(result)
    assert "local-generation" in repr(result)


@with_numpy
def test_pin_protects_mmap_result(tmp_path):
    import gc

    import numpy as np

    memory = make_memory(tmp_path, mmap_mode="r")

    @memory.cache
    def f():
        return np.arange(100, dtype="float64")

    result = f.call_and_shelve()
    lease = result.pin()
    loaded = result.get()
    assert isinstance(loaded, np.memmap)
    memory.reduce_size(items_limit=0)
    # The mapping stays valid because the underlying file is protected.
    assert loaded.sum() == np.arange(100).sum()

    # The lease is released only once the mapping is closed, as documented.
    del loaded
    gc.collect()
    lease.release()
    memory.reduce_size(items_limit=0)
    with pytest.raises(KeyError):
        result.get()


def test_session_liveness_is_lock_based_not_pid_based(tmp_path):
    location = str(tmp_path)
    session = _acquire_session(location)
    session_path = os.path.join(
        location, SESSIONS_DIRNAME, "{}.lock".format(session.session_id)
    )
    assert session_is_alive(session_path)
    _close_session(location, session)
    assert not session_is_alive(session_path)
    # A missing session file cannot keep a phantom lease alive.
    assert not session_is_alive(session_path + ".missing")


# ---------------------------------------------------------------------------
# Legacy layout compatibility
# ---------------------------------------------------------------------------
def test_legacy_layout_remains_unchanged_and_rejects_pin(tmp_path):
    memory = Memory(location=str(tmp_path), verbose=0)

    @memory.cache
    def f(x):
        return x

    result = f.call_and_shelve(1)
    assert result.generation is None
    with pytest.raises(RuntimeError, match="generation layout"):
        result.pin()

    for dirpath, dirnames, filenames in os.walk(memory.store_backend.location):
        assert GENERATIONS_DIRNAME not in dirnames
        assert "leases" not in dirnames
        assert "current" not in filenames

    info_path = os.path.join(memory.store_backend.location, "store_backend_info.json")
    with open(info_path, "rb") as fh:
        assert json.loads(fh.read().decode())["cache_version"] == 2

    # Legacy reduce_size still returns the report structure.
    report = memory.reduce_size(items_limit=0)
    assert report["protected"] == []
    assert report["final_items"] == 0


def test_generation_backend_rejects_legacy_store_with_guidance(tmp_path):
    memory = Memory(location=str(tmp_path), verbose=0)

    @memory.cache
    def f(x):
        return x

    f(1)
    with pytest.raises(ValueError, match="migrat"):
        Memory(location=str(tmp_path), verbose=0, backend=GENERATION_BACKEND)


def test_legacy_backend_rejects_generation_store(tmp_path):
    make_memory(tmp_path)
    with pytest.raises(ValueError, match="generation layout"):
        Memory(location=str(tmp_path), verbose=0, backend="local")


def test_migrate_legacy_store_and_pin_migrated_item(tmp_path):
    legacy = Memory(location=str(tmp_path), verbose=0)

    @legacy.cache
    def f(x):
        return x + 100

    f(1)
    f(2)

    n_migrated = Memory.migrate_to_generation_layout(str(tmp_path))
    assert n_migrated == 2

    memory = Memory(location=str(tmp_path), verbose=0, backend=GENERATION_BACKEND)
    assert isinstance(memory.store_backend, GenerationFileSystemStoreBackend)
    cached = memory.cache(f)
    assert cached.check_call_in_cache(1)
    result = cached.call_and_shelve(1)
    assert result.get() == 101
    assert result.generation == "gen-000001"
    lease = result.pin()
    memory.reduce_size(items_limit=0)
    assert result.get() == 101
    lease.release()


def test_migration_rejects_old_cache_tree(tmp_path):
    legacy = Memory(location=str(tmp_path), verbose=0)

    @legacy.cache
    def f(x):
        return x

    f(1)
    info_path = os.path.join(legacy.store_backend.location, "store_backend_info.json")
    with open(info_path, "w") as fh:
        json.dump({"cache_version": 1, "require_update": True}, fh)

    with pytest.raises(ValueError, match="update_cache_tree"):
        Memory.migrate_to_generation_layout(str(tmp_path))


def test_migration_is_idempotent_on_generation_store(tmp_path):
    make_memory(tmp_path)
    assert Memory.migrate_to_generation_layout(str(tmp_path)) == 0


# ---------------------------------------------------------------------------
# Two-process tests with explicit file barriers
# ---------------------------------------------------------------------------
_PIN_SCRIPT = r"""
import json
import os
import sys
import time

from joblib import Memory, MemorizedResult

LOCATION, BARRIER, ROLE = sys.argv[1], sys.argv[2], sys.argv[3]


def wait_for(name, timeout=60):
    path = os.path.join(BARRIER, name)
    deadline = time.time() + timeout
    while not os.path.exists(path):
        time.sleep(0.02)
        if time.time() > deadline:
            raise RuntimeError("timed out waiting for barrier " + name)


def touch(name):
    with open(os.path.join(BARRIER, name), "w") as fh:
        fh.write("")


def f(x):
    return x * 10


if ROLE == "setup":
    mem = Memory(location=LOCATION, backend="local-generation", verbose=0)
    result = mem.cache(f).call_and_shelve(7)
    with open(os.path.join(BARRIER, "args.json"), "w") as fh:
        json.dump(
            {
                "location": mem.store_backend.location,
                "call_id": list(result._call_id),
                "generation": result.generation,
            },
            fh,
        )
    touch("setup-done")

elif ROLE == "pinner-hold":
    name = sys.argv[4] if len(sys.argv) > 4 else ""
    suffix = "-{}".format(name) if name else ""
    mem = Memory(location=LOCATION, backend="local-generation", verbose=0)
    result = mem.cache(f).call_and_shelve(7)
    expected = result.get()
    lease = result.pin()
    touch("pinned{}".format(suffix))
    wait_for("release{}".format(suffix))
    assert result.get() == expected == 70
    lease.release()
    touch("done{}".format(suffix))

elif ROLE == "evictor":
    wait_for("start")
    mem = Memory(location=LOCATION, backend="local-generation", verbose=0)
    mem.reduce_size(items_limit=0)
    touch("evicted")

elif ROLE == "acquire-fail":
    wait_for("evicted")
    with open(os.path.join(BARRIER, "args.json")) as fh:
        args = json.loads(fh.read())
    result = MemorizedResult(
        args["location"],
        tuple(args["call_id"]),
        backend="local-generation",
        generation=args["generation"],
    )
    try:
        result.pin()
    except KeyError:
        pass
    else:
        raise AssertionError("pinning an evicted generation must fail")
    leases_dir = os.path.join(
        result.store_backend._item_path(result._call_id), "leases"
    )
    assert not os.path.exists(leases_dir) or not os.listdir(leases_dir)
    touch("acquire-failed")

elif ROLE == "hold-forever":
    mem = Memory(location=LOCATION, backend="local-generation", verbose=0)
    result = mem.cache(f).call_and_shelve(7)
    lease = result.pin()  # noqa: F841
    touch("pinned")
    while not os.path.exists(os.path.join(BARRIER, "kill")):
        time.sleep(0.05)

elif ROLE == "round-pinner":
    rounds = int(sys.argv[4])
    mem = Memory(location=LOCATION, backend="local-generation", verbose=0)
    cached = mem.cache(f)
    for i in range(rounds):
        wait_for("pin-{}".format(i))
        result = cached.call_and_shelve(7)
        lease = result.pin()
        assert result.get() == 70
        touch("pinned-{}".format(i))
        wait_for("release-{}".format(i))
        lease.release()
        touch("released-{}".format(i))

elif ROLE == "round-evictor":
    rounds = int(sys.argv[4])
    mem = Memory(location=LOCATION, backend="local-generation", verbose=0)
    for i in range(rounds):
        wait_for("go-{}".format(i))
        mem.reduce_size(items_limit=0)
        touch("evicted-{}".format(i))
"""


def _run_role(role, location, barrier, *extra_args):
    env = os.environ.copy()
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")])
    cmd = [sys.executable, "-c", _PIN_SCRIPT, location, barrier, role]
    cmd.extend(extra_args)
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_barrier(barrier, name, timeout=60):
    path = os.path.join(barrier, name)
    deadline = time.time() + timeout
    while not os.path.exists(path):
        time.sleep(0.02)
        if time.time() > deadline:
            raise AssertionError("timed out waiting for " + name)


def _touch(barrier, name):
    with open(os.path.join(barrier, name), "w") as fh:
        fh.write("")


def _wait_process(proc, timeout=60):
    stdout, stderr = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, "subprocess failed:\n{}".format(stderr)
    return stdout, stderr


def test_two_processes_pin_wins_over_eviction(tmp_path):
    location = str(tmp_path / "store")
    barrier = str(tmp_path / "barrier")
    os.makedirs(barrier)

    pinner = _run_role("pinner-hold", location, barrier)
    _wait_barrier(barrier, "pinned")

    evictor = _run_role("evictor", location, barrier)
    _touch(barrier, "start")
    _wait_process(evictor)
    _wait_barrier(barrier, "evicted")

    # The eviction ran while the lease was held: verify with real files.
    memory = Memory(location=location, verbose=0, backend=GENERATION_BACKEND)
    items = memory.store_backend.get_items()
    assert len(items) == 1
    item_path = items[0].path
    current = memory.store_backend._read_current(item_path)
    surviving_output = os.path.join(
        memory.store_backend._generation_path(item_path, current),
        "output.pkl",
    )
    assert os.path.exists(surviving_output)
    with open(surviving_output, "rb") as fh:
        assert numpy_pickle.load(fh) == 70

    _touch(barrier, "release")
    _wait_process(pinner)
    _wait_barrier(barrier, "done")


def test_two_processes_eviction_wins_acquire_fails_without_window(tmp_path):
    location = str(tmp_path / "store")
    barrier = str(tmp_path / "barrier")
    os.makedirs(barrier)

    _wait_process(_run_role("setup", location, barrier))
    _wait_barrier(barrier, "setup-done")
    with open(os.path.join(barrier, "args.json")) as fh:
        args = json.loads(fh.read())
    pinned_output = os.path.join(
        args["location"],
        *args["call_id"][:-1],
        # split layout: 3-char prefix then the remaining hash
        args["call_id"][-1][:3],
        args["call_id"][-1][3:],
        GENERATIONS_DIRNAME,
        args["generation"],
        "output.pkl",
    )
    assert os.path.exists(pinned_output)

    evictor = _run_role("evictor", location, barrier)
    _touch(barrier, "start")
    _wait_process(evictor)
    _wait_barrier(barrier, "evicted")
    assert not os.path.exists(pinned_output)

    acquirer = _run_role("acquire-fail", location, barrier)
    _wait_process(acquirer)
    _wait_barrier(barrier, "acquire-failed")


def test_pinner_process_killed_releases_protection(tmp_path):
    location = str(tmp_path / "store")
    barrier = str(tmp_path / "barrier")
    os.makedirs(barrier)

    holder = _run_role("hold-forever", location, barrier)
    try:
        _wait_barrier(barrier, "pinned")
        memory = Memory(location=location, verbose=0, backend=GENERATION_BACKEND)
        items = memory.store_backend.get_items()
        assert len(items) == 1
        item_path = items[0].path

        # Eviction cannot delete while the independent owner is alive.
        report = memory.reduce_size(items_limit=0)
        assert report["protected"] == [item_path]
        assert os.path.isdir(item_path)

        holder.terminate()
        holder.wait(timeout=30)

        # The kernel has released the session lock: a later sweep reclaims
        # the stale lease and the generation is deleted for real.
        report = memory.reduce_size(items_limit=0)
        assert report["protected"] == []
        assert not os.path.exists(item_path)
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=30)


def test_pid_reuse_cannot_keep_a_dead_lease_alive(tmp_path):
    location = str(tmp_path / "store")
    memory = Memory(location=location, verbose=0, backend=GENERATION_BACKEND)

    @memory.cache
    def f(x):
        return x

    result = f.call_and_shelve(1)
    item_path = memory.store_backend._item_path(result._call_id)
    leases_dir = os.path.join(item_path, "leases")

    # Forge a marker whose PID is *alive* (this test process) but whose
    # session lock is held by nobody: liveness must come from the kernel
    # lock, never from PID inspection.
    fake_marker = os.path.join(leases_dir, "fake-lease.json")
    os.makedirs(leases_dir, exist_ok=True)
    with open(fake_marker, "w") as fh:
        json.dump(
            {
                "lease_id": "fake-lease",
                "session_id": "dead-session-{}".format(os.getpid()),
                "generation": result.generation,
                "pid": os.getpid(),
                "created_at": time.time(),
            },
            fh,
        )

    memory.reduce_size(items_limit=0)
    assert not os.path.exists(fake_marker), "stale marker was not swept"
    with pytest.raises(KeyError):
        result.get()


def test_alternating_two_process_barrier_rounds(tmp_path):
    # Deterministic race coverage in both orderings, several rounds each.
    location = str(tmp_path / "store")
    barrier = str(tmp_path / "barrier")
    os.makedirs(barrier)
    rounds = 10

    pinner = _run_role("round-pinner", location, barrier, str(rounds))
    evictor = _run_role("round-evictor", location, barrier, str(rounds))
    try:
        for i in range(rounds):
            if i % 2 == 0:
                # Pin first, then evict: protection must hold.
                _touch(barrier, "pin-{}".format(i))
                _wait_barrier(barrier, "pinned-{}".format(i))
                _touch(barrier, "go-{}".format(i))
                _wait_barrier(barrier, "evicted-{}".format(i))
                memory = Memory(
                    location=location,
                    verbose=0,
                    backend=GENERATION_BACKEND,
                )
                assert memory.store_backend.get_items()
                _touch(barrier, "release-{}".format(i))
                _wait_barrier(barrier, "released-{}".format(i))
            else:
                # Evict first, then pin: acquisition recreates the item.
                _touch(barrier, "go-{}".format(i))
                _wait_barrier(barrier, "evicted-{}".format(i))
                _touch(barrier, "pin-{}".format(i))
                _wait_barrier(barrier, "pinned-{}".format(i))
                memory = Memory(
                    location=location,
                    verbose=0,
                    backend=GENERATION_BACKEND,
                )
                assert memory.store_backend.get_items()
                _touch(barrier, "release-{}".format(i))
                _wait_barrier(barrier, "released-{}".format(i))
    finally:
        _wait_process(pinner)
        _wait_process(evictor)

    # Everything released: a final eviction empties the store.
    memory = Memory(location=location, verbose=0, backend=GENERATION_BACKEND)
    memory.reduce_size(items_limit=0)
    assert memory.store_backend.get_items() == []


def test_two_processes_two_independent_owners(tmp_path):
    location = str(tmp_path / "store")
    barrier = str(tmp_path / "barrier")
    os.makedirs(barrier)

    owner_a = _run_role("pinner-hold", location, barrier, "A")
    owner_b = _run_role("pinner-hold", location, barrier, "B")
    _wait_barrier(barrier, "pinned-A")
    _wait_barrier(barrier, "pinned-B")

    memory = Memory(location=location, verbose=0, backend=GENERATION_BACKEND)
    backend = memory.store_backend

    def pinned_generations_on_disk():
        # Protection is verified through real files and values, never through
        # the mere presence of lease markers.
        pinned = []
        for item_path in backend._iter_item_paths(backend.location):
            leases_dir = backend._manager.leases_dir(item_path)
            if not os.path.isdir(leases_dir):
                continue
            for name in os.listdir(leases_dir):
                with open(os.path.join(leases_dir, name)) as fh:
                    marker = json.loads(fh.read())
                gen_path = backend._generation_path(item_path, marker["generation"])
                output = os.path.join(gen_path, "output.pkl")
                assert os.path.exists(output)
                with open(output, "rb") as fh:
                    assert numpy_pickle.load(fh) == 70
                pinned.append(output)
        return pinned

    pinned_files = pinned_generations_on_disk()
    assert len(pinned_files) == 2

    # One owner gone: its generation becomes collectable, the other owner's
    # files must still be physically present and loadable.
    _touch(barrier, "release-A")
    _wait_process(owner_a)
    _wait_barrier(barrier, "done-A")
    memory.reduce_size(items_limit=0)
    surviving = pinned_generations_on_disk()
    assert len(surviving) == 1
    assert surviving[0] in pinned_files

    _touch(barrier, "release-B")
    _wait_process(owner_b)
    _wait_barrier(barrier, "done-B")

    # Both owners gone: normal eviction finally reclaims everything.
    memory.reduce_size(items_limit=0)
    assert memory.store_backend.get_items() == []
    assert pinned_generations_on_disk() == []


def test_concurrent_threads_pin_and_evict(tmp_path):
    import threading

    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x

    result = f.call_and_shelve(1)
    errors = []

    def pin_owner():
        try:
            local_result = result
            for _ in range(20):
                try:
                    lease = local_result.pin()
                except KeyError:
                    # The generation was evicted while no lease was held:
                    # recompute and pin the fresh one.
                    local_result = f.call_and_shelve(1)
                    lease = local_result.pin()
                assert local_result.get() == 1
                lease.release()
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    def evictor():
        try:
            for _ in range(20):
                memory.reduce_size(items_limit=0)
        except BaseException as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [
        threading.Thread(target=pin_owner),
        threading.Thread(target=pin_owner),
        threading.Thread(target=evictor),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors

    # Everything was released at the end: the store can be fully evicted.
    memory.reduce_size(items_limit=0)
    assert memory.store_backend.get_items() == []


def test_validation_callback_invalidation_keeps_pinned_generation(tmp_path):
    memory = make_memory(tmp_path)
    accept = {"value": True}

    @memory.cache(cache_validation_callback=lambda metadata: accept["value"])
    def f(x):
        return x

    result = f.call_and_shelve(1)
    generation = result.generation
    lease = result.pin()

    accept["value"] = False
    f(1)  # callback rejects the old result: logical clear then recompute
    assert result.get() == 1  # pinned generation still readable
    new_result = f.call_and_shelve(1)
    assert new_result.generation != generation
    assert generation in list_generations(memory.store_backend, result._call_id)
    lease.release()


def test_memorized_result_clear_keeps_pinned_files(tmp_path):
    memory = make_memory(tmp_path)

    @memory.cache
    def f(x):
        return x + 1

    result = f.call_and_shelve(1)
    generation = result.generation
    lease = result.pin()
    pinned_path = output_pkl_path(memory.store_backend, result._call_id, generation)

    result.clear()  # logical invalidation of the single item
    assert not memory.store_backend.contains_item(result._call_id)
    assert os.path.exists(pinned_path)
    assert result.get() == 2
    lease.release()
    memory.reduce_size(items_limit=0)
    assert not os.path.exists(pinned_path)
