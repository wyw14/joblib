"""Cross-process primitives for pinning cache generations.

This module implements the lease mechanism used by the generation-layout
filesystem store (see :class:`joblib._store_backends.GenerationFileSystemStoreBackend`).

Design principles
-----------------
* Mutual exclusion between lease acquisition and eviction is provided by a
  store-wide :class:`StoreLock` built on kernel file locks (``fcntl.flock`` on
  POSIX, ``msvcrt.locking`` on Windows). The kernel releases these locks
  automatically when a process exits or is killed, so a crashed owner can never
  deadlock the store: there is no stale-lock reclamation protocol and, in
  particular, no decision is ever made from PID liveness.

* A lease owner registers a *session* file on which it holds an exclusive
  kernel lock for as long as it owns leases. Whether the owner of a lease is
  still alive is answered by trying to acquire that lock non-blockingly:
  failure means "alive", success means "dead". A PID is recorded in the
  session file for diagnostics only. PID reuse can therefore neither extend
  the lifetime of a dead owner's protection nor cut a live owner's lease.

* Lease markers are plain JSON files written atomically (temporary file plus
  ``os.replace``). They only reference the session id and the pinned
  generation, so recovery never depends on ``atexit`` handlers running in the
  owning process: stale markers are swept lazily by the next operation taking
  the store lock.
"""

import json
import os
import threading
import time
import uuid

from .backports import concurrency_safe_rename
from .disk import mkdirp

# Name of the store-wide lock and of the session directory, relative to the
# store location. They are reserved and must never collide with function
# identifier paths.
LOCK_NAME = "_pin.lock"
SESSIONS_DIRNAME = "_sessions"
LEASES_DIRNAME = "leases"

# Timeout (in seconds) when waiting on the store-wide lock. Critical sections
# only perform short filesystem metadata operations, so this is a safety net
# rather than a tuning parameter.
LOCK_TIMEOUT = 30.0
LOCK_RETRY_INTERVAL = 0.05


if os.name == "nt":  # Windows
    import msvcrt

    def _try_lock(fd):
        """Acquire an exclusive non-blocking lock on the first byte."""
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:  # POSIX
    import fcntl

    def _try_lock(fd):
        """Acquire an exclusive non-blocking lock on the file."""
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


class StoreLockTimeoutError(TimeoutError):
    """Raised when the store-wide lock cannot be acquired in time."""


class StoreLock:
    """A cross-process exclusive lock backed by a kernel file lock.

    The lock is automatically released by the kernel if the holding process
    dies, even on ``SIGKILL`` / ``TerminateProcess``. As a consequence there
    is no such thing as a stale store lock and no PID-based recovery.
    """

    def __init__(self, store_location, timeout=LOCK_TIMEOUT):
        self._lock_path = os.path.join(store_location, LOCK_NAME)
        self._timeout = timeout
        self._fd = None

    def __enter__(self):
        mkdirp(os.path.dirname(self._lock_path))
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            # Windows byte-range locks need at least one locked byte.
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
        except OSError:
            # Best effort: another process may have grown the file already.
            pass

        deadline = time.monotonic() + self._timeout
        while True:
            try:
                _try_lock(fd)
                self._fd = fd
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise StoreLockTimeoutError(
                        "Could not acquire the store lock at {} within {}s".format(
                            self._lock_path, self._timeout
                        )
                    )
                time.sleep(LOCK_RETRY_INTERVAL)

    def __exit__(self, exc_type, exc_value, traceback):
        fd, self._fd = self._fd, None
        try:
            _unlock(fd)
        finally:
            os.close(fd)


def session_is_alive(session_path):
    """Return True iff a live process holds the session lock.

    The check is based solely on kernel lock ownership: acquiring the lock
    means the owning process is gone (or released everything), while failing
    to acquire it means a live owner still holds it. The PID recorded in the
    session file is never inspected.
    """
    try:
        fd = os.open(session_path, os.O_RDWR)
    except FileNotFoundError:
        return False
    except OSError:
        # The file exists but cannot be opened: be conservative and assume a
        # live owner is still touching it.
        return True
    try:
        try:
            _try_lock(fd)
        except OSError:
            return True
        # No live holder: drop the probe lock immediately.
        try:
            _unlock(fd)
        except OSError:
            pass
        return False
    finally:
        os.close(fd)


class _Session:
    """A process-owned session holding an exclusive lock on its file."""

    __slots__ = ("session_id", "fd", "leases")

    def __init__(self, session_id, fd):
        self.session_id = session_id
        self.fd = fd
        self.leases = set()


# Process-wide registry of open sessions, keyed by store location. The kernel
# lock provides cross-process safety; this lock only serializes threads.
_REGISTRY_LOCK = threading.Lock()
_SESSIONS = {}


def _session_path(store_location, session_id):
    return os.path.join(store_location, SESSIONS_DIRNAME, "{}.lock".format(session_id))


def _acquire_session(store_location):
    """Return the live session for this process and store, creating it."""
    with _REGISTRY_LOCK:
        session = _SESSIONS.get(store_location)
        if session is not None:
            return session

        session_id = uuid.uuid4().hex
        mkdirp(os.path.join(store_location, SESSIONS_DIRNAME))
        path = _session_path(store_location, session_id)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        payload = json.dumps(
            {
                "session_id": session_id,
                # Diagnostic only: liveness never relies on this PID.
                "pid": os.getpid(),
                "created_at": time.time(),
            }
        ).encode("utf-8")
        os.write(fd, payload)
        # A brand-new unique file: the lock must succeed immediately.
        _try_lock(fd)

        session = _Session(session_id, fd)
        _SESSIONS[store_location] = session
        return session


def _close_session(store_location, session):
    try:
        _unlock(session.fd)
    except OSError:
        pass
    try:
        os.close(session.fd)
    except OSError:
        pass
    try:
        os.remove(_session_path(store_location, session.session_id))
    except OSError:
        pass


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    mkdirp(directory)
    tmp = os.path.join(
        directory, ".{}.{}.tmp".format(os.path.basename(path), uuid.uuid4().hex)
    )
    with open(tmp, "wb") as fh:
        fh.write(json.dumps(payload).encode("utf-8"))
    concurrency_safe_rename(tmp, path)


class ResultLease:
    """Handle on a pinned cache generation.

    Instances are context managers and support nesting (several owners can
    pin the same generation). :meth:`release` is idempotent. A lease must be
    held for as long as the pinned value is in use, in particular when it has
    been loaded with ``mmap_mode``: once every lease on a generation has been
    released, ordinary eviction rules may delete its files again.
    """

    def __init__(self, manager, store_location, item_path, lease_id, generation):
        self._manager = manager
        self._store_location = store_location
        self._item_path = item_path
        self._lease_id = lease_id
        self._generation = generation
        self._released = False

    @property
    def lease_id(self):
        return self._lease_id

    @property
    def generation(self):
        return self._generation

    @property
    def item_path(self):
        return self._item_path

    @property
    def leased(self):
        return not self._released

    def release(self):
        """Release the lease. Safe to call more than once."""
        if self._released:
            return
        self._manager._release(self)
        self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def __del__(self):
        # Best-effort cleanup for forgotten releases. This is only a
        # convenience: cross-process safety does not rely on it, nor on
        # atexit, since the kernel releases the session lock on process exit.
        try:
            self.release()
        except Exception:
            pass

    def __repr__(self):
        return "ResultLease(generation={!r}, leased={})".format(
            self._generation, self.leased
        )


def _output_is_complete(gen_path, timeout=10.0):
    """Return True iff the generation's output.pkl can really be opened.

    Opening the file (rather than testing os.path.exists, which on Windows
    returns False on transient sharing violations) makes pin acquisition
    robust against antivirus scans and concurrent replacements. Only a
    missing file reports a non-existing generation.
    """
    filename = os.path.join(gen_path, "output.pkl")
    deadline = time.monotonic() + timeout
    while True:
        try:
            with open(filename, "rb"):
                return True
        except FileNotFoundError:
            return False
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


# Sentinel: an item has markers whose liveness could not be established (for
# instance a transient sharing violation while reading a marker). Eviction
# then conservatively protects the whole item rather than risking a forced
# deletion of genuinely pinned content.
PROTECT_ALL = object()


class LeaseManager:
    """Acquire, release and recover generation leases for one store."""

    def __init__(self, store_location):
        self.location = store_location

    def lock(self):
        return StoreLock(self.location)

    @staticmethod
    def leases_dir(item_path):
        return os.path.join(item_path, LEASES_DIRNAME)

    def _marker_path(self, item_path, lease_id):
        return os.path.join(self.leases_dir(item_path), "{}.json".format(lease_id))

    @staticmethod
    def _read_marker(marker):
        """Read a marker, retrying transient sharing violations.

        Returns the parsed payload, None if the marker disappeared, or
        ``PROTECT_ALL`` if it cannot be read/parsed within the retry budget:
        the caller must then keep the protection on the safe side.
        """
        deadline = time.monotonic() + 10.0
        while True:
            try:
                with open(marker, "rb") as fh:
                    return json.loads(fh.read().decode("utf-8"))
            except FileNotFoundError:
                return None
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    return PROTECT_ALL
                time.sleep(0.05)

    def sweep_leases(self, item_path):
        """Remove markers proven stale. Call under the store lock.

        Returns a set of still-protected generation ids. When the liveness
        of a marker cannot be established, :data:`PROTECT_ALL` is added so
        that every generation of the item is conservatively kept. Markers
        are only ever removed after positively proving that their owner
        session is dead, never on a transient read error.
        """
        leases_dir = self.leases_dir(item_path)
        active = set()
        try:
            names = os.listdir(leases_dir)
        except FileNotFoundError:
            return active
        except OSError:
            # Cannot even inspect the leases directory: keep everything.
            active.add(PROTECT_ALL)
            return active

        for name in names:
            if not name.endswith(".json"):
                continue
            marker = os.path.join(leases_dir, name)
            data = self._read_marker(marker)
            if data is None:
                # Removed concurrently by another sweeper.
                continue
            if data is PROTECT_ALL:
                active.add(PROTECT_ALL)
                continue
            try:
                session_id = data["session_id"]
                generation = data["generation"]
            except (KeyError, TypeError):
                active.add(PROTECT_ALL)
                continue

            if session_is_alive(_session_path(self.location, session_id)):
                active.add(generation)
            else:
                # Positive proof of death (no holder on the session lock):
                # only now is it safe to drop the marker.
                self._remove_file(marker)
                self._remove_file(_session_path(self.location, session_id))
        return active

    def all_active_generations(self):
        """Sweep every item in the store. Call under the store lock.

        Returns a mapping ``item_path -> set of protected generation ids``.
        Items mapped to :data:`PROTECT_ALL` must be protected wholesale.
        """
        protected = {}
        for dirpath, dirnames, _ in os.walk(self.location):
            if os.path.basename(dirpath) == LEASES_DIRNAME:
                item_path = os.path.dirname(dirpath)
                active = self.sweep_leases(item_path)
                if active:
                    protected[item_path] = active
                # No lease can be nested below the leases directory.
                dirnames[:] = []
        return protected

    def acquire(self, item_path, generation):
        """Atomically pin a complete generation.

        Either the generation is complete and a lease marker is created inside
        the same critical section that eviction has to take, or the generation
        is gone and a :class:`KeyError` is raised. There is no observable
        window in which the files exist but are unprotected.
        """
        gen_path = os.path.join(item_path, "generations", generation)
        with self.lock():
            if not _output_is_complete(gen_path):
                raise KeyError(
                    "Cannot pin a non-existing cache generation: {}".format(gen_path)
                )

            # Housekeeping: drop markers of dead owners before deciding that
            # the generation is protected.
            self.sweep_leases(item_path)

            session = _acquire_session(self.location)
            lease_id = uuid.uuid4().hex
            _atomic_write_json(
                self._marker_path(item_path, lease_id),
                {
                    "lease_id": lease_id,
                    "session_id": session.session_id,
                    "generation": generation,
                    "pid": os.getpid(),
                    "created_at": time.time(),
                },
            )
            session.leases.add(lease_id)
            return ResultLease(self, self.location, item_path, lease_id, generation)

    def _release(self, lease):
        with self.lock():
            self._remove_file(self._marker_path(lease._item_path, lease._lease_id))
            with _REGISTRY_LOCK:
                session = _SESSIONS.get(lease._store_location)
                if session is not None:
                    session.leases.discard(lease._lease_id)
                    if not session.leases:
                        _close_session(lease._store_location, session)
                        del _SESSIONS[lease._store_location]

    @staticmethod
    def _remove_file(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            # On Windows a file can be transiently locked by another process
            # (e.g. an antivirus scan). A later sweep will retry the removal.
            pass
