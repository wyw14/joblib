"""Storage providers backends for Memory caching."""

import collections
import datetime
import json
import operator
import os
import os.path
import pathlib
import shutil
import threading
import time
import types
import uuid
import warnings
from abc import ABCMeta, abstractmethod
from pickle import PicklingError

from . import numpy_pickle
from .backports import concurrency_safe_rename
from .disk import memstr_to_bytes, mkdirp, rm_subdirs
from .logger import format_time

CacheItemInfo = collections.namedtuple("CacheItemInfo", "path size last_access")


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
        """
        items_to_delete = self._get_items_to_delete(bytes_limit, items_limit, age_limit)

        for item in items_to_delete:
            if self.verbose > 10:
                print("Deleting item {0}".format(item))
            try:
                self.clear_location(item.path)
            except OSError:
                # Even with ignore_errors=True shutil.rmtree can raise OSError
                # with:
                # [Errno 116] Stale file handle if another process has deleted
                # the folder already.
                pass

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


@_split_decorator
class FileSystemStoreBackend(StoreBackendBase, StoreBackendMixin):
    """A StoreBackend used with local or network file systems."""

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
            with open(info_path, "rb") as file:
                info = json.loads(file.read().decode("utf-8"))
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
            with open(info_path, "wb") as file:
                file.write(json.dumps(info).encode("utf-8"))

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
        with open(info_path, "wb") as file:
            file.write(json.dumps(info).encode("utf-8"))

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
        with open(info_path, "wb") as file:
            file.write(json.dumps(info).encode("utf-8"))

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


def _normalize_layer_path(path):
    """Return a normalized, absolute path with symlinks resolved, for
    comparing layer locations independently of spelling/links."""
    if isinstance(path, pathlib.Path):
        path = str(path)
    path = os.path.expanduser(path)
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _validate_layer_locations(primary_location, fallback_locations):
    """Validate the layered store configuration.

    Duplicate locations and locations overlapping one another (one being the
    same directory or nested inside the other, including through symlinks)
    are rejected with a ``ValueError``.
    """
    locations = [primary_location] + list(fallback_locations)
    normalized = []
    for location in locations:
        normalized_location = _normalize_layer_path(location)
        for existing in normalized:
            if normalized_location == existing:
                raise ValueError(
                    "The same location is configured twice in a layered "
                    "store backend: %r" % location
                )
            if normalized_location.startswith(
                existing + os.sep
            ) or existing.startswith(normalized_location + os.sep):
                raise ValueError(
                    "Layered store backends cannot use overlapping or "
                    "nested locations: %r and %r"
                    % (location, existing)
                )
        normalized.append(normalized_location)
    return normalized


def _build_read_only_layer(location, verbose, compress, mmap_mode):
    """Configure a FileSystemStoreBackend on an existing directory without
    ever writing into it (no directory creation, no store_backend_info.json
    write, no .gitignore write), so that genuinely read-only mounts can be
    used as fallback layers."""
    backend = FileSystemStoreBackend()
    backend.location = location
    backend.compress = compress
    backend.mmap_mode = mmap_mode
    backend.verbose = verbose

    cache_version = 2
    info_path = os.path.join(location, "store_backend_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "rb") as info_file:
                info = json.loads(info_file.read().decode("utf-8"))
            cache_version = info.get("cache_version", 2)
        except (OSError, ValueError):
            # Unreadable or corrupted info file: default to the current
            # cache tree and rely on the layer's items being well-formed.
            pass
    if cache_version == 1:
        backend._split_id = types.MethodType(_old_split_id, backend)
    return backend


def _reconstruct_layered_store_backend(
    location, fallback_locations, promote, verbose, compress, mmap_mode
):
    obj = LayeredFileSystemStoreBackend()
    obj.configure(
        location,
        verbose=verbose,
        backend_options=dict(
            fallback=list(fallback_locations),
            promote=promote,
            compress=compress,
            mmap_mode=mmap_mode,
        ),
    )
    return obj


class LayeredFileSystemStoreBackend(StoreBackendBase):
    """A filesystem-only store backend combining several cache directories.

    The backend is made of one writable *primary* layer and an ordered list
    of *read-only fallback* layers. Lookups first query the primary layer and
    then, in order, each fallback layer. Writes (new computations, metadata,
    function code, clears and size reductions) always target the primary
    layer only: fallback directories are never modified nor deleted.

    A cache hit is only served from a layer when that layer alone holds a
    *complete* candidate for the call: a ``func_code.py`` matching the
    wrapped function, together with the call's ``output.pkl`` and a valid
    ``metadata.json``. A single call is never assembled from pieces coming
    from different layers.

    Parameters understood by ``configure`` through ``backend_options``:

    fallback: str, path-like or list of them, optional
        Ordered read-only fallback cache directories. Missing directories
        are skipped (with a warning) instead of being created.
    promote: bool, optional
        When True, a candidate loaded from a fallback layer is copied into
        the primary layer. The whole candidate directory is first copied to
        a primary-private temporary location and then atomically published,
        so an interrupted promotion can never leave a half candidate (only
        an output or only metadata) visible in the primary layer.
    compress, mmap_mode: same options as FileSystemStoreBackend.
    """

    _open_item = staticmethod(open)
    _item_exists = staticmethod(os.path.exists)
    _move_item = staticmethod(concurrency_safe_rename)

    # Name of the primary-private directory holding in-flight promotions.
    # It lives outside of any function/call tree, so its contents can never
    # be mistaken for a cache candidate.
    _promotion_tmp_dirname = ".promotion_tmp"

    def configure(self, location, verbose=1, backend_options=None):
        """Configure the layered store.

        ``location`` is the writable primary layer. ``backend_options`` may
        contain ``fallback`` (ordered read-only locations) and ``promote``
        (boolean, defaults to False) alongside the usual ``compress`` and
        ``mmap_mode`` options.
        """
        if backend_options is None:
            backend_options = {}

        fallback = backend_options.get("fallback", [])
        if fallback is None:
            fallback = []
        if isinstance(fallback, (str, pathlib.Path)):
            fallback = [fallback]
        fallback = [
            str(p) if isinstance(p, pathlib.Path) else p for p in fallback
        ]

        self.promote = bool(backend_options.get("promote", False))
        self.compress = backend_options.get("compress", False)
        self.mmap_mode = backend_options.get("mmap_mode")
        self.verbose = verbose

        if self.compress and self.mmap_mode is not None:
            warnings.warn(
                "Compressed items cannot be memmapped in a "
                "filesystem store. Option will be ignored.",
                stacklevel=2,
            )

        # Reject duplicate, overlapping or symlink-nested configurations
        # before any directory is created.
        _validate_layer_locations(location, fallback)

        # The primary layer is a regular, writable filesystem store.
        self.primary = FileSystemStoreBackend()
        self.primary.configure(
            location,
            verbose=verbose,
            backend_options=dict(
                compress=self.compress, mmap_mode=self.mmap_mode
            ),
        )
        self.location = self.primary.location

        # Fallback layers are read-only: they are configured without any
        # write attempt, and directories that do not exist are skipped
        # rather than created.
        fallbacks = []
        for fallback_location in fallback:
            if not os.path.isdir(fallback_location):
                warnings.warn(
                    "Fallback cache directory %r does not exist; ignoring "
                    "it. No directory was created." % fallback_location,
                    UserWarning,
                    stacklevel=2,
                )
                continue
            fallbacks.append(
                _build_read_only_layer(
                    fallback_location,
                    verbose=verbose,
                    compress=self.compress,
                    mmap_mode=self.mmap_mode,
                )
            )
        self.fallbacks = tuple(fallbacks)
        # Immutable after configuration: every resolution iterates this
        # tuple with local variables, there is no shared "current layer".
        self._layers = (self.primary,) + self.fallbacks

        # Per-call locks serializing in-process promotions. Cross-process
        # safety comes from unique temporary names and atomic renames.
        self._promotion_locks = {}
        self._promotion_locks_guard = threading.Lock()

        # Best-effort cleanup of promotions interrupted by a hard crash.
        # These leftovers are never visible as candidates.
        tmp_root = os.path.join(
            self.primary.location, self._promotion_tmp_dirname
        )
        if os.path.isdir(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ------------------------------------------------------------------
    # Candidate resolution. Every method below is stateless and thread
    # safe: the selected layer and all derived paths are local variables.
    # ------------------------------------------------------------------
    def _item_dir(self, layer, call_id):
        """Real on-disk call directory of ``call_id`` in ``layer``."""
        return os.path.join(layer.location, *layer._split_id(call_id))

    def _reference_func_code(self, func_id):
        """Return the stored function code of the highest-priority layer
        that has one, or None if no layer carries function code."""
        for layer in self._layers:
            filename = os.path.join(layer.location, func_id, "func_code.py")
            try:
                with open(filename, "rb") as f:
                    return f.read().decode("utf-8")
            except OSError:
                continue
        return None

    def _is_complete_candidate(self, layer, call_id, reference_func_code):
        """Check that ``layer`` alone holds a complete candidate.

        The candidate must contain, inside this single layer:
        - a ``func_code.py`` identical to the reference (itself validated
          against the wrapped function by MemorizedFunc),
        - a loadable ``metadata.json`` dictionary,
        - an ``output.pkl`` file.
        """
        item_dir = self._item_dir(layer, call_id)
        output_filename = os.path.join(item_dir, "output.pkl")
        metadata_filename = os.path.join(item_dir, "metadata.json")
        if not (
            os.path.isfile(output_filename)
            and os.path.isfile(metadata_filename)
        ):
            return False

        try:
            with open(metadata_filename, "rb") as f:
                metadata = json.loads(f.read().decode("utf-8"))
        except (OSError, ValueError):
            return False
        if not isinstance(metadata, dict):
            return False

        func_code_filename = os.path.join(
            layer.location, call_id[0], "func_code.py"
        )
        try:
            with open(func_code_filename, "rb") as f:
                layer_func_code = f.read().decode("utf-8")
        except OSError:
            return False
        return layer_func_code == reference_func_code

    def _resolve_candidate(self, call_id):
        """Return ``(layer, item_dir)`` of the first complete candidate,
        in layer priority order, or ``(None, None)`` on a full miss."""
        for layer, item_dir in self.iter_candidates(call_id):
            return layer, item_dir
        return None, None

    def iter_candidates(self, call_id):
        """Yield ``(layer, item_dir)`` complete candidates in priority order.

        The generator is stateless: iterating it never mutates any shared
        attribute, so concurrent threads can resolve different calls (and
        different layers) independently.
        """
        reference_func_code = self._reference_func_code(call_id[0])
        if reference_func_code is None:
            return
        for layer in tuple(self._layers):
            if self._is_complete_candidate(
                layer, call_id, reference_func_code
            ):
                yield layer, self._item_dir(layer, call_id)

    def read_metadata(self, item_dir):
        """Read metadata from an explicitly resolved candidate directory."""
        try:
            with open(os.path.join(item_dir, "metadata.json"), "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except (OSError, ValueError):
            return {}

    def invalidate_candidate(self, call_id, layer):
        """Invalidate a candidate rejected by the validation callback.

        Only the writable primary layer can be modified. A rejected
        read-only fallback candidate is merely skipped and its directory is
        never deleted nor altered.
        """
        if layer is self.primary:
            self.primary.clear_item(call_id)

    # ------------------------------------------------------------------
    # Read operations: always resolve a complete candidate first.
    # ------------------------------------------------------------------
    def contains_item(self, call_id):
        """Check if a complete candidate exists in any layer."""
        layer, _ = self._resolve_candidate(call_id)
        return layer is not None

    def get_metadata(self, call_id):
        """Return the metadata of the resolved candidate, or {} on miss."""
        layer, item_dir = self._resolve_candidate(call_id)
        if layer is None:
            return {}
        try:
            with open(os.path.join(item_dir, "metadata.json"), "rb") as f:
                return json.loads(f.read().decode("utf-8"))
        except (OSError, ValueError):
            return {}

    def get_item_info(self, call_id):
        """Return the real location of the resolved candidate."""
        _, item_dir = self._resolve_candidate(call_id)
        if item_dir is None:
            item_dir = self._item_dir(self.primary, call_id)
        return {"location": item_dir}

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        """Load an item from the layer holding its complete candidate.

        With ``mmap_mode`` set, the memmap is bound to the real file of the
        selected layer. No global "current layer" state is involved, so
        concurrent threads resolving different layers cannot be served the
        wrong file.
        """
        layer, item_dir = self._resolve_candidate(call_id)
        if layer is None:
            missing_filename = os.path.join(
                self._item_dir(self.primary, call_id), "output.pkl"
            )
            raise KeyError(
                "Non-existing item (may have been "
                "cleared).\nFile %s does not exist" % missing_filename
            )
        return self._load_from(
            layer, item_dir, call_id, verbose, timestamp, metadata
        )

    def load_resolved_item(
        self, resolved, call_id, verbose=1, timestamp=None, metadata=None
    ):
        """Load the exact candidate previously selected by
        :meth:`iter_candidates` (e.g. accepted by a validation callback),
        without re-running the selection against a possibly changed store.

        ``resolved`` is a ``(layer, item_dir)`` tuple. The same candidate
        identity rules as :meth:`load_item` apply.
        """
        layer, item_dir = resolved
        return self._load_from(
            layer, item_dir, call_id, verbose, timestamp, metadata
        )

    def _load_from(
        self, layer, item_dir, call_id, verbose=1, timestamp=None, metadata=None
    ):
        """Load ``output.pkl`` from an explicitly selected layer/directory."""
        if verbose > 1:
            ts_string = (
                "{: <16}".format(format_time(time.time() - timestamp))
                if timestamp is not None
                else ""
            )
            signature = os.path.basename(call_id[0])
            if metadata is not None and "input_args" in metadata:
                kwargs = ", ".join(
                    "{}={}".format(*item)
                    for item in metadata["input_args"].items()
                )
                signature += "({})".format(kwargs)
            msg = "[Memory]{}: Loading {}".format(ts_string, signature)
            if verbose < 10:
                print("{0}...".format(msg))
            else:
                print("{0} from {1}".format(msg, item_dir))

        filename = os.path.join(item_dir, "output.pkl")
        if self.mmap_mode is None:
            with layer._open_item(filename, "rb") as f:
                item = numpy_pickle.load(f)
        else:
            # Bind the memmap to the concrete, selected file location.
            item = numpy_pickle.load(filename, mmap_mode=self.mmap_mode)

        if self.promote and layer is not self.primary:
            self._promote(call_id, layer, item_dir)
        return item

    def get_cached_func_code(self, call_id):
        """Return the function code shared by the highest-priority layer
        having one, preserving the single-backend contract of raising on
        absence (MemorizedFunc then (re)writes the code in the primary)."""
        func_code = self._reference_func_code(call_id[0])
        if func_code is None:
            raise FileNotFoundError(
                os.path.join(self.primary.location, *call_id, "func_code.py")
            )
        return func_code

    def get_cached_func_info(self, call_id):
        """Return the real function directory of the first layer holding
        it, falling back to the primary location."""
        for layer in self._layers:
            func_path = os.path.join(layer.location, *call_id)
            if os.path.exists(func_path):
                return {"location": func_path}
        return {"location": os.path.join(self.primary.location, *call_id)}

    def contains_path(self, call_id):
        """Check the cached function path in any layer."""
        for layer in self._layers:
            if os.path.exists(os.path.join(layer.location, *call_id)):
                return True
        return False

    # ------------------------------------------------------------------
    # Write and deletion operations: primary layer only. Fallback layers
    # are never written to, cleared or reduced.
    # ------------------------------------------------------------------
    def dump_item(self, call_id, item, verbose=1):
        self.primary.dump_item(call_id, item, verbose=verbose)

    def store_metadata(self, call_id, metadata):
        self.primary.store_metadata(call_id, metadata)

    def store_cached_func_code(self, call_id, func_code=None):
        self.primary.store_cached_func_code(call_id, func_code)

    def clear_item(self, call_id):
        """Clear the item in the primary layer only. A read-only fallback
        candidate for the same call is left untouched and remains readable
        on the next lookup."""
        self.primary.clear_item(call_id)

    def clear_path(self, call_id):
        """Clear the function path in the primary layer only. Read-only
        fallback layers are never modified."""
        self.primary.clear_path(call_id)

    def create_location(self, location):
        self.primary.create_location(location)

    def clear_location(self, location):
        self.primary.clear_location(location)

    def get_items(self):
        """Return the primary layer items only. Fallback items are not
        accounted for, hence never deleted by size reduction."""
        return self.primary.get_items()

    def clear(self):
        """Clear the primary layer only. Read-only fallback layers keep
        their contents and can still produce cache hits afterwards."""
        self.primary.clear()

    def enforce_store_limits(self, bytes_limit, items_limit=None, age_limit=None):
        """Enforce store limits on the primary layer only."""
        self.primary.enforce_store_limits(
            bytes_limit, items_limit=items_limit, age_limit=age_limit
        )

    # ------------------------------------------------------------------
    # Promotion of a fallback candidate into the primary layer.
    # ------------------------------------------------------------------
    def _get_promotion_lock(self, call_id):
        with self._promotion_locks_guard:
            lock = self._promotion_locks.get(call_id)
            if lock is None:
                lock = threading.Lock()
                self._promotion_locks[call_id] = lock
            return lock

    def _promote_func_code(self, func_id, source_layer):
        """Publish the function code into the primary layer first, using
        its atomic write, so the primary never lacks code identity once a
        promoted call directory is visible."""
        primary_func_code = os.path.join(
            self.primary.location, func_id, "func_code.py"
        )
        if os.path.exists(primary_func_code):
            return
        source_func_code = os.path.join(
            source_layer.location, func_id, "func_code.py"
        )
        with open(source_func_code, "rb") as f:
            func_code = f.read().decode("utf-8")
        self.primary.store_cached_func_code([func_id], func_code)

    def _promote(self, call_id, source_layer, source_item_dir):
        """Copy a complete fallback candidate to the primary layer.

        The complete candidate directory is copied to a primary-private
        unique temporary location and then atomically renamed to its final
        location: the target therefore only ever appears as a whole,
        complete candidate. On failure the temporary copy is removed and
        the untouched fallback candidate keeps serving the call.
        """
        lock = self._get_promotion_lock(call_id)
        with lock:
            reference_func_code = self._reference_func_code(call_id[0])
            if self._is_complete_candidate(
                self.primary, call_id, reference_func_code
            ):
                # Another thread or process already promoted it.
                return

            target_item_dir = self._item_dir(self.primary, call_id)
            if os.path.exists(target_item_dir):
                # Defensive: an incomplete target must not shadow the
                # complete fallback candidate.
                shutil.rmtree(target_item_dir, ignore_errors=True)

            tmp_root = os.path.join(
                self.primary.location, self._promotion_tmp_dirname
            )
            tmp_item_dir = os.path.join(
                tmp_root,
                "{}-{}-{}".format(
                    uuid.uuid4().hex,
                    os.getpid(),
                    id(threading.current_thread()),
                ),
            )
            try:
                # 1. Function code identity first (atomic file publish).
                self._promote_func_code(call_id[0], source_layer)
                # 2. Copy the whole candidate: output.pkl and metadata.json
                #    travel together inside one directory.
                mkdirp(tmp_root)
                shutil.copytree(source_item_dir, tmp_item_dir)
                # 3. Single atomic directory publication. The target path
                #    cannot appear with only one of the two files.
                mkdirp(os.path.dirname(target_item_dir))
                self._move_item(tmp_item_dir, target_item_dir)
                # Remove the now-empty, primary-private temporary root.
                try:
                    os.rmdir(tmp_root)
                except OSError:
                    pass
            except Exception as e:  # noqa: E722
                shutil.rmtree(tmp_item_dir, ignore_errors=True)
                try:
                    os.rmdir(tmp_root)
                except OSError:
                    pass
                warnings.warn(
                    "Unable to promote cached item from read-only fallback "
                    "layer %r to the primary layer. The untouched fallback "
                    "candidate continues to be used. Exception: %s."
                    % (source_layer.location, e),
                    CacheWarning,
                    stacklevel=2,
                )

    def __repr__(self):
        fallback_locations = ", ".join(
            layer.location for layer in self.fallbacks
        )
        return (
            '{class_name}(location="{location}", '
            'fallback=[{fallback}], promote={promote})'.format(
                class_name=self.__class__.__name__,
                location=self.location,
                fallback=fallback_locations,
                promote=self.promote,
            )
        )

    def __reduce__(self):
        return (
            _reconstruct_layered_store_backend,
            (
                self.primary.location,
                [layer.location for layer in self.fallbacks],
                self.promote,
                self.verbose,
                self.compress,
                self.mmap_mode,
            ),
        )
