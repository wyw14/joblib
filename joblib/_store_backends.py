"""Storage providers backends for Memory caching."""

import collections
import datetime
import json
import operator
import os
import os.path
import shutil
import threading
import time
import types
import uuid
import warnings
from abc import ABCMeta, abstractmethod
from pickle import PicklingError

from . import numpy_pickle
from ._pinning import PROTECT_ALL, LeaseManager, _output_is_complete
from .backports import concurrency_safe_rename
from .disk import memstr_to_bytes, mkdirp, rm_subdirs
from .logger import format_time

CacheItemInfo = collections.namedtuple("CacheItemInfo", "path size last_access")

# Generation layout constants. The generation layout is opt-in: it is only
# used by stores whose ``store_backend_info.json`` carries
# ``"layout": "generation"`` and opened through the dedicated backend.
GENERATION_LAYOUT = "generation"
GENERATION_BACKEND_NAME = "local-generation"
GENERATION_CACHE_VERSION = 3
GENERATIONS_DIRNAME = "generations"
CURRENT_FILENAME = "current"


class CacheWarning(Warning):
    """Warning to capture dump failures except for PicklingError."""

    pass


def concurrency_safe_write(object_to_write, filename, write_func):
    """Writes an object into a unique file in a concurrency-safe way."""
    # Temporary name is composed of UUID, process_id and thread_id to avoid
    # collisions due to concurrent write.
    # UUID is unique across nodes and time and help avoid collisions, even if
    # the cache folder is shared by several Python processes with the same pid and
    # thread id on different nodes of a cluster for instance.
    thread_id = id(threading.current_thread())
    temporary_filename = f"{filename}.{uuid.uuid4().hex}-{os.getpid()}-{thread_id}"

    write_func(object_to_write, temporary_filename)

    return temporary_filename


class StoreBackendBase(metaclass=ABCMeta):
    """Helper Abstract Base Class which defines all methods that
    a StorageBackend must implement."""

    location = None

    @abstractmethod
    def _open_item(self, f, mode):
        """Opens an item on the store and return a file-like object.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        f: a file-like object
            The file-like object where an item is stored and retrieved
        mode: string, optional
            the mode in which the file-like object is opened allowed valued are
            'rb', 'wb'

        Returns
        -------
        a file-like object
        """

    @abstractmethod
    def _item_exists(self, location):
        """Checks if an item location exists in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        location: string
            The location of an item. On a filesystem, this corresponds to the
            absolute path, including the filename, of a file.

        Returns
        -------
        True if the item exists, False otherwise
        """

    @abstractmethod
    def _move_item(self, src, dst):
        """Moves an item from src to dst in the store.

        This method is private and only used by the StoreBackendMixin object.

        Parameters
        ----------
        src: string
            The source location of an item
        dst: string
            The destination location of an item
        """

    @abstractmethod
    def create_location(self, location):
        """Creates a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory.
        """

    @abstractmethod
    def clear_location(self, location):
        """Clears a location on the store.

        Parameters
        ----------
        location: string
            The location in the store. On a filesystem, this corresponds to a
            directory or a filename absolute path
        """

    @abstractmethod
    def get_items(self):
        """Returns the whole list of items available in the store.

        Returns
        -------
        The list of items identified by their ids (e.g filename in a
        filesystem).
        """

    @abstractmethod
    def configure(self, location, verbose=0, backend_options=dict()):
        """Configures the store.

        Parameters
        ----------
        location: string
            The base location used by the store. On a filesystem, this
            corresponds to a directory.
        verbose: int
            The level of verbosity of the store
        backend_options: dict
            Contains a dictionary of named parameters used to configure the
            store backend.
        """


class StoreBackendMixin(object):
    """Class providing all logic for managing the store in a generic way.

    The StoreBackend subclass has to implement 3 methods: create_location,
    clear_location and configure. The StoreBackend also has to provide
    a private _open_item, _item_exists and _move_item methods. The _open_item
    method has to have the same signature as the builtin open and return a
    file-like object.
    """

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        """Load an item from the store given its id as a list of str."""
        full_path = os.path.join(self.location, *call_id)

        if verbose > 1:
            ts_string = (
                "{: <16}".format(format_time(time.time() - timestamp))
                if timestamp is not None
                else ""
            )
            signature = os.path.basename(call_id[0])
            if metadata is not None and "input_args" in metadata:
                kwargs = ", ".join(
                    "{}={}".format(*item) for item in metadata["input_args"].items()
                )
                signature += "({})".format(kwargs)
            msg = "[Memory]{}: Loading {}".format(ts_string, signature)
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, full_path))

        mmap_mode = None if not hasattr(self, "mmap_mode") else self.mmap_mode

        filename = os.path.join(full_path, "output.pkl")
        if not self._item_exists(filename):
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % filename
            )

        # file-like object cannot be used when mmap_mode is set
        if mmap_mode is None:
            with self._open_item(filename, "rb") as f:
                item = numpy_pickle.load(f)
        else:
            item = numpy_pickle.load(filename, mmap_mode=mmap_mode)
        return item

    def dump_item(self, call_id, item, verbose=1):
        """Dump an item in the store at the id given as a list of str."""
        try:
            item_path = os.path.join(self.location, *call_id)
            if not self._item_exists(item_path):
                self.create_location(item_path)
            filename = os.path.join(item_path, "output.pkl")
            if verbose > 10:
                print("Persisting in %s" % item_path)

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, "wb") as f:
                    try:
                        numpy_pickle.dump(to_write, f, compress=self.compress)
                    except PicklingError as e:
                        # TODO(1.5) turn into error
                        warnings.warn(
                            "Unable to cache to disk: failed to pickle "
                            "output. In version 1.5 this will raise an "
                            f"exception. Exception: {e}.",
                            FutureWarning,
                        )

            self._concurrency_safe_write(item, filename, write_func)
        except Exception as e:  # noqa: E722
            warnings.warn(
                "Unable to cache to disk. Possibly a race condition in the "
                f"creation of the directory. Exception: {e}.",
                CacheWarning,
            )

    def clear_item(self, call_id):
        """Clear the item at the id, given as a list of str."""
        item_path = os.path.join(self.location, *call_id)
        if self._item_exists(item_path):
            self.clear_location(item_path)

    def contains_item(self, call_id):
        """Check if there is an item at the id, given as a list of str."""
        item_path = os.path.join(self.location, *call_id)
        filename = os.path.join(item_path, "output.pkl")

        return self._item_exists(filename)

    def get_item_info(self, call_id):
        """Return information about item."""
        return {"location": os.path.join(self.location, *call_id)}

    def get_metadata(self, call_id):
        """Return actual metadata of an item."""
        try:
            item_path = os.path.join(self.location, *call_id)
            filename = os.path.join(item_path, "metadata.json")
            with self._open_item(filename, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except:  # noqa: E722
            return {}

    def store_metadata(self, call_id, metadata):
        """Store metadata of a computation."""
        try:
            item_path = os.path.join(self.location, *call_id)
            self.create_location(item_path)
            filename = os.path.join(item_path, "metadata.json")

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, "wb") as f:
                    f.write(json.dumps(to_write).encode("utf-8"))

            self._concurrency_safe_write(metadata, filename, write_func)
        except:  # noqa: E722
            pass

    def contains_path(self, call_id):
        """Check cached function is available in store."""
        func_path = os.path.join(self.location, *call_id)
        return self._item_exists(func_path)

    def clear_path(self, call_id):
        """Clear all items with a common path in the store."""
        func_path = os.path.join(self.location, *call_id)
        if self._item_exists(func_path):
            self.clear_location(func_path)

    def store_cached_func_code(self, call_id, func_code=None):
        """Store the code of the cached function."""
        func_path = os.path.join(self.location, *call_id)
        if not self._item_exists(func_path):
            self.create_location(func_path)

        if func_code is not None:
            filename = os.path.join(func_path, "func_code.py")

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, "wb") as f:
                    f.write(to_write.encode("utf-8"))

            # Written in place, another process could read it half-written and
            # take that for a changed function, wiping the cache (#1694)
            self._concurrency_safe_write(func_code, filename, write_func)

    def get_cached_func_code(self, call_id):
        """Store the code of the cached function."""
        filename = os.path.join(self.location, *call_id, "func_code.py")
        try:
            with self._open_item(filename, "rb") as f:
                return f.read().decode("utf-8")
        except:  # noqa: E722
            raise

    def get_cached_func_info(self, call_id):
        """Return information related to the cached function if it exists."""
        return {"location": os.path.join(self.location, *call_id)}

    def clear(self):
        """Clear the whole store content."""
        self.clear_location(self.location)

    def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Remove the store's oldest files to enforce item, byte, and age limits.

        Returns a report describing what was deleted and, when pinning is
        supported by the backend, which limits could not be met because of
        pinned generations.
        """
        requested_bytes_limit = (
            memstr_to_bytes(bytes_limit)
            if isinstance(bytes_limit, str)
            else bytes_limit
        )
        age_limit_seconds = age_limit.total_seconds() if age_limit is not None else None

        initial_items = self.get_items()
        initial_size = sum(item.size for item in initial_items)

        items_to_delete = self._get_items_to_delete(bytes_limit, items_limit, age_limit)

        deleted = []
        for item in items_to_delete:
            if self.verbose > 10:
                print("Deleting item {0}".format(item))
            try:
                self.clear_location(item.path)
                deleted.append(item.path)
            except OSError:
                # Even with ignore_errors=True shutil.rmtree can raise OSError
                # with:
                # [Errno 116] Stale file handle if another process has deleted
                # the folder already.
                pass

        final_items = self.get_items()
        final_size = sum(item.size for item in final_items)

        return {
            "bytes_limit": requested_bytes_limit,
            "items_limit": items_limit,
            "age_limit_seconds": age_limit_seconds,
            "initial_size": initial_size,
            "final_size": final_size,
            "initial_items": len(initial_items),
            "final_items": len(final_items),
            "deleted": deleted,
            # The legacy layout has no pinning support so these are always
            # empty/zero: every selected item is deleted unconditionally.
            "protected": [],
            "age_unmet": [],
            "pinned_bytes": 0,
            "unmet_bytes": (
                max(0, final_size - requested_bytes_limit)
                if requested_bytes_limit is not None
                else 0
            ),
            "unmet_items": (
                max(0, len(final_items) - items_limit) if items_limit is not None else 0
            ),
        }

    def _get_items_to_delete(self, bytes_limit, items_limit=None, age_limit=None):
        """
        Get items to delete to keep the store under size, file, & age limits.
        """
        if isinstance(bytes_limit, str):
            bytes_limit = memstr_to_bytes(bytes_limit)

        items = self.get_items()
        if not items:
            return []

        size = sum(item.size for item in items)

        if bytes_limit is not None:
            to_delete_size = size - bytes_limit
        else:
            to_delete_size = 0

        if items_limit is not None:
            to_delete_items = len(items) - items_limit
        else:
            to_delete_items = 0

        if age_limit is not None:
            older_item = min(item.last_access for item in items)
            if age_limit.total_seconds() < 0:
                raise ValueError("age_limit has to be a positive timedelta")
            deadline = datetime.datetime.now() - age_limit
        else:
            deadline = None

        if (
            to_delete_size <= 0
            and to_delete_items <= 0
            and (deadline is None or older_item > deadline)
        ):
            return []

        # We want to delete first the cache items that were accessed a
        # long time ago
        items.sort(key=operator.attrgetter("last_access"))

        items_to_delete = []
        size_so_far = 0
        items_so_far = 0

        for item in items:
            if (
                (size_so_far >= to_delete_size)
                and items_so_far >= to_delete_items
                and (deadline is None or deadline < item.last_access)
            ):
                break

            items_to_delete.append(item)
            size_so_far += item.size
            items_so_far += 1

        return items_to_delete

    def _concurrency_safe_write(self, to_write, filename, write_func):
        """Writes an object into a file in a concurrency-safe way."""
        temporary_filename = concurrency_safe_write(to_write, filename, write_func)
        self._move_item(temporary_filename, filename)

    def __repr__(self):
        """Printable representation of the store location."""
        return '{class_name}(location="{location}")'.format(
            class_name=self.__class__.__name__, location=self.location
        )


def _check_hex(s, length):
    if len(s) != length:
        return False
    try:
        if length % 2 == 1:
            # bytes.fromhex waits for pair sequences
            s += "0"
        bytes.fromhex(s)
        return True
    except ValueError:
        return False


def _old_split_id(self, call_id):
    info_path = os.path.join(self.location, "store_backend_info.json")
    try:
        with open(info_path, "rb") as file:
            info = json.loads(file.read().decode("utf-8"))
        if info["cache_version"] == 2:
            self._split_id = types.MethodType(FileSystemStoreBackend._split_id, self)
            return self._split_id(call_id)
    except OSError:
        pass
    return call_id


def _reconstructStoreBackend(cls, location, verbose, compress, mmap_mode):
    obj = cls.__new__(cls)
    obj.configure(location, verbose, dict(compress=compress, mmap_mode=mmap_mode))
    return obj


def _replace_cache_dir(old_dir, new_dir):
    """
    Move an old cache directory to a newer one depending on
    the last modification time of output.pkl
    """
    old_item = os.path.join(old_dir, "output.pkl")
    try:
        old_mtime = os.path.getmtime(old_item)
    except OSError:
        # "output.pkl" does not exist, so we don't replace
        shutil.rmtree(old_dir)
        return
    new_item = os.path.join(new_dir, "output.pkl")
    try:
        new_mtime = os.path.getmtime(new_item)
    except OSError:
        new_mtime = -1
    if new_mtime < old_mtime:
        # replace
        shutil.rmtree(new_dir)
        os.replace(old_dir, new_dir)
    else:
        # don't replace
        shutil.rmtree(old_dir)


def _split_decorator(cls):
    """Helper to split call_id in all relevant class methods.

    Use a decorator as all methods are not implemented directly
    in this class.
    """

    def split_method_decorator(method):
        def split_method(self, call_id, *args, **kwargs):
            call_id = self._split_id(call_id)
            return method(self, call_id, *args, **kwargs)

        return split_method

    for method in [
        "load_item",
        "dump_item",
        "clear_item",
        "contains_item",
        "get_item_info",
        "get_metadata",
        "store_metadata",
    ]:
        setattr(cls, method, split_method_decorator(getattr(StoreBackendMixin, method)))
    return cls


def _write_store_info(info_path, info):
    """Atomically write the store information file."""
    tmp = "{}.{}.tmp".format(info_path, uuid.uuid4().hex)
    with open(tmp, "wb") as fh:
        fh.write(json.dumps(info).encode("utf-8"))
    _robust_atomic_replace(tmp, info_path)


def _robust_atomic_replace(src, dst, timeout=10.0, retry_interval=0.05):
    """Replace dst with src, retrying transient Windows sharing violations."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            concurrency_safe_rename(src, dst)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(retry_interval)


def _read_store_info(info_path, retries=5, delay=0.05):
    """Read the store information file, tolerating concurrent atomic writes.

    A concurrent writer replacing the file can briefly expose an empty read;
    retrying turns this into a deterministic success instead of a JSON parse
    failure.
    """
    for attempt in range(retries):
        try:
            with open(info_path, "rb") as fh:
                content = fh.read()
            return json.loads(content.decode("utf-8"))
        except (OSError, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _write_gitignore(location):
    """Automatically add a ``.gitignore`` file to the cache folder."""
    # XXX: the condition is necessary because in `Memory.__init__`, the user
    # passed `location` param is modified to be either `{location}` or
    # `{location}/joblib` depending on input type (`pathlib.Path` vs `str`).
    # The proper resolution of this inconsistency is tracked in:
    # https://github.com/joblib/joblib/issues/1684
    cache_directory = (
        os.path.dirname(location)
        if os.path.dirname(location) and os.path.basename(location) == "joblib"
        else location
    )
    gitignore = os.path.join(cache_directory, ".gitignore")
    if not os.path.exists(gitignore):
        try:
            with open(gitignore, "w") as file:
                file.write("# Created by joblib automatically.\n")
                file.write("*\n")
        except OSError as e:
            warnings.warn(f"Unable to write {gitignore}. Exception: {e}.")


@_split_decorator
class FileSystemStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""

    name = "local"
    generation_layout = False

    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    def clear_location(self, location):
        """Delete location on store."""
        if location == self.location:
            rm_subdirs(location)
        else:
            shutil.rmtree(location, ignore_errors=True)

    def create_location(self, location):
        """Create object location on store"""
        mkdirp(location)

    def get_items(self):
        """Returns the whole list of items available in the store."""
        items = []

        for dirpath, _, filenames in os.walk(self.location):
            parent, basename = os.path.split(dirpath)
            if not _check_hex(basename, 32):
                if not _check_hex(basename, 29):
                    continue
                if not _check_hex(os.path.basename(parent), 3):
                    continue

            output_filename = os.path.join(dirpath, "output.pkl")
            try:
                last_access = os.path.getatime(output_filename)
            except OSError:
                try:
                    last_access = os.path.getatime(dirpath)
                except OSError:
                    # The directory has already been deleted
                    continue

            last_access = datetime.datetime.fromtimestamp(last_access)
            try:
                full_filenames = [os.path.join(dirpath, fn) for fn in filenames]
                dirsize = sum(os.path.getsize(fn) for fn in full_filenames)
            except OSError:
                # Either output_filename or one of the files in
                # dirpath does not exist any more. We assume this
                # directory is being cleaned by another process already
                continue

            items.append(CacheItemInfo(dirpath, dirsize, last_access))

        return items

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend.

        For this backend, valid store options are 'compress' and 'mmap_mode'
        """
        if backend_options is None:
            backend_options = {}

        # setup location directory
        self.location = location
        info = None
        if not os.path.exists(self.location):
            mkdirp(self.location)
            info = {"cache_version": 2, "require_update": False}

        # Get info
        info_path = os.path.join(self.location, "store_backend_info.json")
        if os.path.exists(info_path):
            info = _read_store_info(info_path)
            if info.get("layout") == GENERATION_LAYOUT:
                raise ValueError(
                    "The store at '{}' uses the generation layout, which is "
                    "only supported by the '{}' store backend. Instantiate "
                    "Memory(..., backend='{}') to open it.".format(
                        self.location,
                        GENERATION_BACKEND_NAME,
                        GENERATION_BACKEND_NAME,
                    )
                )
        else:
            if info is None:
                # Cache directory without info.
                # As long as it has subdirectories,
                # we consider it uses an old cache tree
                with os.scandir(self.location) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            info = {"cache_version": 1, "require_update": True}
                            break
                if info is None:
                    info = {"cache_version": 2, "require_update": False}
            _write_store_info(info_path, info)

        # Warn if an update is required
        if info["require_update"]:
            true_location = (
                f"'{os.path.dirname(location)}'"
                if os.path.basename(location) == "joblib"
                else f"pathlib.Path('{location}')"
            )
            warnings.warn(
                f"The FileSystemStoreBackend at '{self.location}' may contains "
                "items using old cache storage tree.\n"
                "The joblib cache tree has recently been updated "
                "for efficiency reasons.\n"
                "Starting with joblib 1.9, the old cache tree "
                "will no longer be supported.\n"
                f"Please run `joblib.Memory({true_location}).update_cache_tree()` "
                "to update your cache tree."
            )

        # Using old split method for older cache version
        if info["cache_version"] == 1:
            self._split_id = types.MethodType(_old_split_id, self)

        # Automatically add `.gitignore` file to the cache folder.
        _write_gitignore(location)

        # item can be stored compressed for faster I/O
        self.compress = backend_options.get("compress", False)

        # FileSystemStoreBackend can be used with mmap_mode options under
        # certain conditions.
        mmap_mode = backend_options.get("mmap_mode")
        if self.compress and mmap_mode is not None:
            warnings.warn(
                "Compressed items cannot be memmapped in a "
                "filesystem store. Option will be ignored.",
                stacklevel=2,
            )

        self.mmap_mode = mmap_mode
        self.verbose = verbose

    def _split_id(self, call_id):
        if len(call_id) == 0 or not _check_hex(call_id[-1], 32):
            return call_id
        return (*call_id[:-1], call_id[-1][:3], call_id[-1][3:])

    # XXX: To remove in joblib 1.9
    def update_cache_tree(self):
        # First info update
        info_path = os.path.join(self.location, "store_backend_info.json")
        info = {"cache_version": 2, "require_update": True}
        _write_store_info(info_path, info)

        # Find old cache directories
        old_cache = []
        for dirpath, _, files in os.walk(self.location):
            if not _check_hex(os.path.basename(dirpath), 32):
                continue
            if not (("output.pkl" in files) or ("metadata.json" in files)):
                continue
            old_cache.append(dirpath)

        # Replace old cache directories
        for dirpath in old_cache:
            parent, basename = os.path.split(dirpath)
            newdir_parent = os.path.join(parent, basename[:3])
            newdir = os.path.join(newdir_parent, basename[3:])
            if os.path.exists(newdir):
                # If the new cache directory already exists,
                # it is replaced depending on the last modification time
                # of 'output.pkl'
                _replace_cache_dir(dirpath, newdir)
            else:
                os.makedirs(newdir_parent, exist_ok=True)
                os.replace(dirpath, newdir)

        # Second info update
        info["require_update"] = False
        _write_store_info(info_path, info)

    def __reduce__(self):
        return (
            _reconstructStoreBackend,
            (
                self.__class__,
                self.location,
                self.verbose,
                self.compress,
                self.mmap_mode,
            ),
        )


def _generation_store_info():
    return {
        "cache_version": GENERATION_CACHE_VERSION,
        "layout": GENERATION_LAYOUT,
        "require_update": False,
    }


def _directory_is_non_empty(path):
    try:
        return len(os.listdir(path)) > 0
    except OSError:
        return False


def migrate_legacy_store_to_generation_layout(location):
    """Migrate a legacy (cache version 2) store to the generation layout.

    Each legacy item directory is turned into an item with a single
    ``gen-000001`` generation holding its original ``output.pkl`` and
    ``metadata.json``, a ``current`` pointer and an empty ``leases``
    directory. Stores still using the old cache tree (cache version 1) must
    be updated with :meth:`Memory.update_cache_tree` first.

    Returns the number of migrated cache items.
    """
    info_path = os.path.join(location, "store_backend_info.json")
    if os.path.exists(info_path):
        with open(info_path, "rb") as fh:
            info = json.loads(fh.read().decode("utf-8"))
        if info.get("layout") == GENERATION_LAYOUT:
            return 0
        if info.get("cache_version") == 1:
            raise ValueError(
                "The store at '{}' uses the old cache tree "
                "(cache_version=1). Run Memory(location).update_cache_tree() "
                "before migrating to the generation layout.".format(location)
            )

    mkdirp(location)

    # Collect the item directories before moving anything: os.walk must not
    # observe the generations/ directories being created during migration.
    item_dirs = []
    for dirpath, _, filenames in os.walk(location):
        if "output.pkl" in filenames:
            item_dirs.append(dirpath)

    for item_path in item_dirs:
        gen_path = os.path.join(item_path, GENERATIONS_DIRNAME, "gen-000001")
        mkdirp(gen_path)
        for filename in ("output.pkl", "metadata.json"):
            source = os.path.join(item_path, filename)
            if os.path.exists(source):
                os.replace(source, os.path.join(gen_path, filename))

        # Keep the function identity the generation was computed with next to
        # the generation itself.
        parent = item_path
        for _ in range(6):
            parent = os.path.dirname(parent)
            func_code = os.path.join(parent, "func_code.py")
            if os.path.exists(func_code):
                try:
                    shutil.copyfile(func_code, os.path.join(gen_path, "func_code.py"))
                except OSError:
                    pass
                break

        with open(os.path.join(item_path, CURRENT_FILENAME), "wb") as fh:
            fh.write(b"gen-000001")
        mkdirp(os.path.join(item_path, "leases"))

    _write_store_info(info_path, _generation_store_info())

    return len(item_dirs)


class GenerationFileSystemStoreBackend(FileSystemStoreBackend):
    """Filesystem store with immutable generations and cross-process pins.

    Each cached item is stored under immutable, self-contained generations::

        <item_path>/
            current                     # text file: id of the active generation
            generations/
                <generation_id>/
                    output.pkl
                    metadata.json
                    func_code.py        # snapshot of the function identity
            leases/
                <lease_id>.json         # one marker per live owner

    Recomputing a call never overwrites files in place: a brand-new generation
    directory is written and the ``current`` pointer is atomically replaced.
    Old generations referenced by a lease (a *pin*) are physically preserved
    by :meth:`clear_item`, :meth:`clear_path`, :meth:`clear` and
    :meth:`enforce_store_limits`, even though they are logically invalidated.

    The store root additionally reserves ``_pin.lock`` (the cross-process
    critical-section lock) and ``_sessions/`` (owner session lock files).
    """

    name = GENERATION_BACKEND_NAME
    generation_layout = True

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def configure(self, location, verbose=1, backend_options=None):
        """Configure the store backend.

        For this backend, valid store options are 'compress' and 'mmap_mode'.
        A legacy store is never silently converted: opening one raises a
        :class:`ValueError` pointing at the explicit migration helper.
        """
        if backend_options is None:
            backend_options = {}

        self.location = location
        mkdirp(location)

        info_path = os.path.join(location, "store_backend_info.json")
        if os.path.exists(info_path):
            try:
                info = _read_store_info(info_path)
            except (OSError, ValueError):
                # An unreadable marker is either a leftover of a crashed
                # first configuration or an externally corrupted file.
                if self._has_legacy_items(location):
                    raise ValueError(
                        "The store info file at '{}' is unreadable while the "
                        "directory contains legacy cache content. Repair or "
                        "remove it before opening the store with backend "
                        "'{}'.".format(info_path, GENERATION_BACKEND_NAME)
                    )
                info = None
            if info is not None and info.get("layout") != GENERATION_LAYOUT:
                raise ValueError(
                    "The store at '{}' uses the legacy cache layout and "
                    "cannot be opened with backend '{}'. Migrate it "
                    "explicitly with "
                    "Memory.migrate_to_generation_layout(location) "
                    "first.".format(location, GENERATION_BACKEND_NAME)
                )
            if info is None:
                _write_store_info(info_path, _generation_store_info())
        else:
            # Atomic first-initialization claim, concurrent-process safe.
            self._claim_fresh_store(location, info_path)

        self._manager = LeaseManager(location)
        self._thread_local = threading.local()

        _write_gitignore(location)

        self.compress = backend_options.get("compress", False)
        mmap_mode = backend_options.get("mmap_mode")
        if self.compress and mmap_mode is not None:
            warnings.warn(
                "Compressed items cannot be memmapped in a "
                "filesystem store. Option will be ignored.",
                stacklevel=2,
            )
        self.mmap_mode = mmap_mode
        self.verbose = verbose

    @staticmethod
    def _has_legacy_items(location):
        """Detect legacy items without any initialization-time race.

        The generation layout only ever writes ``output.pkl`` below a
        ``generations/`` directory, so an ``output.pkl`` outside one is an
        unambiguous legacy cache item. Function directories or lock files
        created by concurrent first-time configurations are not mistaken for
        legacy content.
        """
        for dirpath, dirnames, filenames in os.walk(location):
            if os.path.basename(dirpath) == GENERATIONS_DIRNAME:
                dirnames[:] = []
                continue
            if "output.pkl" in filenames:
                return True
        return False

    def _claim_fresh_store(self, location, info_path):
        """First-initialization protocol, safe under concurrent processes."""
        try:
            fd = os.open(info_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # Another process won the claim: wait for its marker content.
            info = _read_store_info(info_path)
            if info.get("layout") != GENERATION_LAYOUT:
                raise ValueError(
                    "The store at '{}' uses the legacy cache layout and "
                    "cannot be opened with backend '{}'. Migrate it "
                    "explicitly with "
                    "Memory.migrate_to_generation_layout(location) "
                    "first.".format(location, GENERATION_BACKEND_NAME)
                )
            return

        # Claim owner: nothing else can be initializing this store now.
        try:
            if self._has_legacy_items(location):
                raise ValueError(
                    "The directory '{}' already contains a legacy joblib "
                    "cache. Use "
                    "Memory.migrate_to_generation_layout(location) to "
                    "convert it before opening it with backend '{}'.".format(
                        location, GENERATION_BACKEND_NAME
                    )
                )
            os.close(fd)
            _write_store_info(info_path, _generation_store_info())
        except BaseException:
            # Do not leave an empty claim file behind on failure.
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(info_path)
            except OSError:
                pass
            raise

    def update_cache_tree(self):
        raise RuntimeError(
            "update_cache_tree only applies to the legacy cache tree; the "
            "generation layout ('{}' backend) has no legacy tree to "
            "update.".format(GENERATION_BACKEND_NAME)
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _item_path(self, call_id):
        return os.path.join(self.location, *self._split_id(call_id))

    def _generations_dir(self, item_path):
        return os.path.join(item_path, GENERATIONS_DIRNAME)

    def _generation_path(self, item_path, generation):
        return os.path.join(self._generations_dir(item_path), generation)

    def _current_path(self, item_path):
        return os.path.join(item_path, CURRENT_FILENAME)

    def _read_current(self, item_path):
        # On Windows a concurrent replacement or an antivirus scan can turn
        # an open() into a transient PermissionError while the file still
        # exists. Only FileNotFoundError means "no current generation".
        path = self._current_path(item_path)
        deadline = time.monotonic() + 10.0
        while True:
            try:
                with open(path, "rb") as fh:
                    generation = fh.read().decode("utf-8").strip()
                return generation or None
            except FileNotFoundError:
                return None
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def _write_current(self, item_path, generation):
        # Atomic pointer publication: readers only ever see the previous or
        # the new generation, never a half-written pointer.
        tmp = "{}.{}.tmp".format(self._current_path(item_path), uuid.uuid4().hex)
        with open(tmp, "wb") as fh:
            fh.write(generation.encode("utf-8"))
        _robust_atomic_replace(tmp, self._current_path(item_path))

    def _resolve_generation(self, item_path, generation=None):
        if generation is not None:
            return generation
        return self._read_current(item_path)

    def _pending_generations(self):
        # dump_item and store_metadata run in the same thread for a given
        # computation; a thread-local map pairs them without cross-thread
        # confusion.
        pending = getattr(self._thread_local, "pending", None)
        if pending is None:
            pending = {}
            self._thread_local.pending = pending
        return pending

    def _latest_unpublished_generation(self, item_path):
        """Fallback pairing: newest generation with output but no metadata."""
        generations_dir = self._generations_dir(item_path)
        try:
            names = os.listdir(generations_dir)
        except OSError:
            return None
        candidates = []
        for name in names:
            gen_path = os.path.join(generations_dir, name)
            if not os.path.isdir(gen_path):
                continue
            if os.path.exists(
                os.path.join(gen_path, "output.pkl")
            ) and not os.path.exists(os.path.join(gen_path, "metadata.json")):
                try:
                    candidates.append((os.path.getmtime(gen_path), name))
                except OSError:
                    continue
        if not candidates:
            return None
        return max(candidates)[1]

    @staticmethod
    def _directory_size(path):
        try:
            return sum(
                os.path.getsize(os.path.join(path, name))
                for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name))
            )
        except OSError:
            return 0

    # ------------------------------------------------------------------
    # Item reading
    # ------------------------------------------------------------------
    def get_current_generation(self, call_id):
        """Return the id of the current generation of an item, or None."""
        return self._read_current(self._item_path(call_id))

    def contains_item(self, call_id):
        """Check if the item has a complete current generation."""
        item_path = self._item_path(call_id)
        generation = self._read_current(item_path)
        if generation is None:
            return False
        return _output_is_complete(self._generation_path(item_path, generation))

    def load_item(
        self,
        call_id,
        verbose=1,
        timestamp=None,
        metadata=None,
        generation=None,
    ):
        """Load an item, pinned to a specific generation when given one."""
        item_path = self._item_path(call_id)
        generation = self._resolve_generation(item_path, generation)
        if generation is None:
            raise KeyError(
                "Non-existing item (may have been cleared).\n"
                "No current generation in {}".format(item_path)
            )
        filename = os.path.join(
            self._generation_path(item_path, generation), "output.pkl"
        )

        mmap_mode = None if not hasattr(self, "mmap_mode") else self.mmap_mode
        # Open the file directly instead of testing existence first: on
        # Windows a transient sharing violation (antivirus, concurrent
        # replacement) makes os.path.exists return False, which would be
        # wrongly reported as a missing item. FileNotFoundError is the only
        # reliable "gone" signal; other OSErrors are retried.
        deadline = time.monotonic() + 10.0
        while True:
            try:
                if mmap_mode is None:
                    with self._open_item(filename, "rb") as f:
                        return numpy_pickle.load(f)
                return numpy_pickle.load(filename, mmap_mode=mmap_mode)
            except FileNotFoundError:
                raise KeyError(
                    "Non-existing item (may have been cleared).\n"
                    "File %s does not exist" % filename
                )
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def get_metadata(self, call_id, generation=None):
        """Return the metadata of the current (or a pinned) generation."""
        try:
            item_path = self._item_path(call_id)
            generation = self._resolve_generation(item_path, generation)
            if generation is None:
                return {}
            filename = os.path.join(
                self._generation_path(item_path, generation), "metadata.json"
            )
            with self._open_item(filename, "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except:  # noqa: E722
            return {}

    def get_items(self):
        """List current generations as the logical items of the store."""
        items = []
        for dirpath, dirnames, filenames in os.walk(self.location):
            if CURRENT_FILENAME not in filenames:
                continue
            # Nothing below an item directory can be an item itself.
            dirnames[:] = []
            generation = self._read_current(dirpath)
            if generation is None:
                continue
            gen_path = self._generation_path(dirpath, generation)
            output_filename = os.path.join(gen_path, "output.pkl")
            try:
                last_access = os.path.getatime(output_filename)
            except OSError:
                try:
                    last_access = os.path.getatime(gen_path)
                except OSError:
                    # The generation disappeared concurrently: skip it.
                    continue
            last_access = datetime.datetime.fromtimestamp(last_access)
            dirsize = self._directory_size(gen_path)
            items.append(CacheItemInfo(dirpath, dirsize, last_access))
        return items

    # ------------------------------------------------------------------
    # Item writing: every (re)computation creates a new generation
    # ------------------------------------------------------------------
    def dump_item(self, call_id, item, verbose=1):
        """Write the output into a brand-new, not-yet-published generation."""
        try:
            item_path = self._item_path(call_id)
            generation = "gen-" + uuid.uuid4().hex[:16]
            gen_path = self._generation_path(item_path, generation)
            mkdirp(gen_path)
            filename = os.path.join(gen_path, "output.pkl")
            if verbose > 10:
                print("Persisting in %s" % gen_path)

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, "wb") as f:
                    try:
                        numpy_pickle.dump(to_write, f, compress=self.compress)
                    except PicklingError as e:
                        # TODO(1.5) turn into error
                        warnings.warn(
                            "Unable to cache to disk: failed to pickle "
                            "output. In version 1.5 this will raise an "
                            f"exception. Exception: {e}.",
                            FutureWarning,
                        )

            self._concurrency_safe_write(item, filename, write_func)
            self._pending_generations()[item_path] = generation
        except Exception as e:  # noqa: E722
            warnings.warn(
                "Unable to cache to disk. Possibly a race condition in the "
                f"creation of the directory. Exception: {e}.",
                CacheWarning,
            )

    def store_metadata(self, call_id, metadata):
        """Publish a generation once both its output and metadata are ready."""
        try:
            item_path = self._item_path(call_id)
            pending = self._pending_generations()
            generation = pending.pop(item_path, None)
            if generation is None:
                # Another thread/process may have created the generation:
                # pair with an unpublished one rather than failing silently.
                generation = self._latest_unpublished_generation(item_path)
            if generation is None:
                return

            gen_path = self._generation_path(item_path, generation)
            if not os.path.exists(os.path.join(gen_path, "output.pkl")):
                return

            filename = os.path.join(gen_path, "metadata.json")

            def write_func(to_write, dest_filename):
                with self._open_item(dest_filename, "wb") as f:
                    f.write(json.dumps(to_write).encode("utf-8"))

            self._concurrency_safe_write(metadata, filename, write_func)
            self._snapshot_func_code(call_id, gen_path)

            # Atomic commit point: the generation only becomes visible once
            # output.pkl and metadata.json are both complete. Publishing a
            # new current generation never deletes previous ones: a
            # MemorizedResult created against the previous generation keeps
            # reading it until explicit eviction, and deleting another
            # writer's in-flight staging generation here would race with it.
            # Crash leftovers are reaped by reduce_size/clear once stale.
            self._write_current(item_path, generation)
        except:  # noqa: E722
            pass

    # A generation without metadata.json never reached the commit point. It
    # is only reaped after this grace period, so a concurrent writer still
    # persisting its output can never have its staging directory removed.
    _STALE_STAGING_SECONDS = 60

    def _is_reapable_staging(self, gen_path):
        if os.path.exists(os.path.join(gen_path, "metadata.json")):
            return True
        try:
            age = time.time() - os.path.getmtime(gen_path)
        except OSError:
            return False
        return age > self._STALE_STAGING_SECONDS

    def _snapshot_func_code(self, call_id, gen_path):
        func_code = os.path.join(self.location, call_id[0], "func_code.py")
        try:
            shutil.copyfile(func_code, os.path.join(gen_path, "func_code.py"))
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Pinning
    # ------------------------------------------------------------------
    def acquire_lease(self, call_id, generation):
        """Pin a generation against physical eviction.

        Raises KeyError when no such complete generation exists: acquisition
        is atomic with respect to eviction, so callers either pin a complete
        generation or learn that it is gone.
        """
        if generation is None:
            raise KeyError(
                "No generation to pin for item {}".format(
                    os.path.join(self.location, *call_id)
                )
            )
        return self._manager.acquire(self._item_path(call_id), generation)

    # ------------------------------------------------------------------
    # Logical invalidation with physical pin preservation
    # ------------------------------------------------------------------
    def clear_item(self, call_id):
        """Logically invalidate an item but keep pinned generations."""
        item_path = self._item_path(call_id)
        if not self._item_exists(item_path):
            return
        with self._manager.lock():
            active = self._manager.sweep_leases(item_path)
            self._invalidate_item_locked(item_path, active)

    def clear_path(self, call_id):
        """Logically invalidate every item below a function path."""
        func_path = os.path.join(self.location, *call_id)
        if not self._item_exists(func_path):
            return
        with self._manager.lock():
            for item_path in self._iter_item_paths(func_path):
                active = self._manager.sweep_leases(item_path)
                self._invalidate_item_locked(item_path, active)
            try:
                os.remove(os.path.join(func_path, "func_code.py"))
            except OSError:
                pass
            self._prune_empty_dirs(func_path)

    def clear(self):
        """Logically invalidate the whole store, preserving pinned content."""
        if not self._item_exists(self.location):
            return
        with self._manager.lock():
            for item_path in self._iter_item_paths(self.location):
                active = self._manager.sweep_leases(item_path)
                self._invalidate_item_locked(item_path, active)

            # Function identities are logically wiped too, but snapshots that
            # live inside pinned generations must remain untouched.
            for dirpath, dirnames, filenames in os.walk(self.location):
                if os.path.basename(dirpath) == GENERATIONS_DIRNAME:
                    dirnames[:] = []
                    continue
                if "func_code.py" in filenames:
                    try:
                        os.remove(os.path.join(dirpath, "func_code.py"))
                    except OSError:
                        pass

            self._prune_empty_dirs(self.location, remove_root=False)

    def clear_location(self, location):
        """Raw physical deletion; pin-aware callers use the clear* methods."""
        if location == self.location:
            rm_subdirs(location)
        else:
            shutil.rmtree(location, ignore_errors=True)

    def _iter_item_paths(self, root):
        for dirpath, dirnames, _ in os.walk(root):
            if GENERATIONS_DIRNAME in dirnames:
                yield dirpath
                # An item directory cannot contain another item.
                dirnames[:] = []

    def _invalidate_item_locked(self, item_path, active):
        """Remove the current pointer and every unprotected generation."""
        if PROTECT_ALL in active:
            # Marker liveness could not be established: never risk deleting
            # genuinely pinned content. The next clear/eviction retries.
            return
        try:
            os.remove(self._current_path(item_path))
        except OSError:
            pass
        generations_dir = self._generations_dir(item_path)
        if os.path.isdir(generations_dir):
            for name in os.listdir(generations_dir):
                if name in active:
                    continue
                gen_path = os.path.join(generations_dir, name)
                # Never remove another writer's in-flight staging generation.
                if not self._is_reapable_staging(gen_path):
                    continue
                shutil.rmtree(gen_path, ignore_errors=True)
        self._prune_item_locked(item_path)

    def _reap_item_locked(self, item_path, active):
        """Delete superseded generations that nobody pins anymore."""
        if PROTECT_ALL in active:
            return
        current = self._read_current(item_path)
        generations_dir = self._generations_dir(item_path)
        if os.path.isdir(generations_dir):
            for name in os.listdir(generations_dir):
                if name == current or name in active:
                    continue
                gen_path = os.path.join(generations_dir, name)
                # Never remove another writer's in-flight staging generation.
                if not self._is_reapable_staging(gen_path):
                    continue
                shutil.rmtree(gen_path, ignore_errors=True)
        self._prune_item_locked(item_path)

    def _prune_item_locked(self, item_path):
        if self._read_current(item_path) is not None:
            return
        generations_dir = self._generations_dir(item_path)
        leases_dir = self._manager.leases_dir(item_path)
        if _directory_is_non_empty(generations_dir) or _directory_is_non_empty(
            leases_dir
        ):
            return
        shutil.rmtree(item_path, ignore_errors=True)
        self._prune_empty_dirs(os.path.dirname(item_path), remove_root=False)

    def _prune_empty_dirs(self, root, remove_root=True):
        if not os.path.isdir(root):
            return
        root = os.path.abspath(root)
        store_root = os.path.abspath(self.location)
        for dirpath, _, _ in os.walk(root, topdown=False):
            dirpath = os.path.abspath(dirpath)
            if dirpath == store_root:
                continue
            if dirpath == root and not remove_root:
                continue
            try:
                os.rmdir(dirpath)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Eviction with pin protection and an explicit unmet-target report
    # ------------------------------------------------------------------
    def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
        requested_bytes_limit = (
            memstr_to_bytes(bytes_limit)
            if isinstance(bytes_limit, str)
            else bytes_limit
        )
        age_limit_seconds = age_limit.total_seconds() if age_limit is not None else None

        initial_items = self.get_items()
        initial_size = sum(item.size for item in initial_items)
        candidates = self._get_items_to_delete(bytes_limit, items_limit, age_limit)
        deadline = (
            datetime.datetime.now() - age_limit if age_limit is not None else None
        )

        deleted = []
        protected = []
        age_unmet = []
        pinned_bytes = 0

        with self._manager.lock():
            # One sweep establishes the protection set used for every
            # decision, so eviction can never race against an acquisition.
            protected_map = self._manager.all_active_generations()

            # Reap every item, including ones that are already logically
            # gone: a generation whose last lease was released must be
            # collectable even when it can no longer be an eviction candidate.
            for item_path in self._iter_item_paths(self.location):
                self._reap_item_locked(item_path, protected_map.get(item_path, set()))

            for item_path, generations in protected_map.items():
                if PROTECT_ALL in generations:
                    # Liveness is uncertain: count the whole item footprint.
                    item_size = 0
                    gens_dir = self._generations_dir(item_path)
                    if os.path.isdir(gens_dir):
                        for name in os.listdir(gens_dir):
                            item_size += self._directory_size(
                                os.path.join(gens_dir, name)
                            )
                    pinned_bytes += item_size
                    continue
                for generation in generations:
                    gen_path = self._generation_path(item_path, generation)
                    if os.path.isdir(gen_path):
                        pinned_bytes += self._directory_size(gen_path)

            for item in candidates:
                item_path = item.path
                current = self._read_current(item_path)
                active = protected_map.get(item_path, set())
                if PROTECT_ALL in active or (current is not None and current in active):
                    # The limit cannot be met for this item: never force
                    # delete protected (or uncertainly protected) content.
                    protected.append(item_path)
                    if deadline is not None and item.last_access <= deadline:
                        age_unmet.append(item_path)
                    continue
                self._invalidate_item_locked(item_path, active)
                deleted.append(item_path)

            final_items = self.get_items()

        final_size = sum(item.size for item in final_items)
        return {
            "bytes_limit": requested_bytes_limit,
            "items_limit": items_limit,
            "age_limit_seconds": age_limit_seconds,
            "initial_size": initial_size,
            "final_size": final_size,
            "initial_items": len(initial_items),
            "final_items": len(final_items),
            "deleted": deleted,
            "protected": protected,
            "age_unmet": age_unmet,
            "pinned_bytes": pinned_bytes,
            "unmet_bytes": (
                max(0, final_size + pinned_bytes - requested_bytes_limit)
                if requested_bytes_limit is not None
                else 0
            ),
            "unmet_items": (
                max(0, len(final_items) - items_limit) if items_limit is not None else 0
            ),
        }
