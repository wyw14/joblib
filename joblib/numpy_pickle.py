"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import collections.abc
import io
import os
import pickle
import warnings

from .backports import make_memmap
from .compressor import (
    _COMPRESSORS,
    LZ4_NOT_INSTALLED_ERROR,
    BinaryZlibFile,
    BZ2CompressorWrapper,
    GzipCompressorWrapper,
    LZ4CompressorWrapper,
    LZMACompressorWrapper,
    XZCompressorWrapper,
    ZlibCompressorWrapper,
    lz4,
    register_compressor,
)

# For compatibility with old versions of joblib, we need ZNDArrayWrapper
# to be visible in the current namespace.
from .numpy_pickle_compat import (
    NDArrayWrapper,
    ZNDArrayWrapper,  # noqa: F401
    load_compatibility,
    readPython2Error,
)
from .numpy_pickle_utils import (
    BUFFER_SIZE,
    Pickler,
    Unpickler,
    _ensure_native_byte_order,
    _read_bytes,
    _reconstruct,
    _validate_fileobject_and_memmap,
    _write_fileobject,
)

# Register supported compressors
register_compressor("zlib", ZlibCompressorWrapper())
register_compressor("gzip", GzipCompressorWrapper())
register_compressor("bz2", BZ2CompressorWrapper())
register_compressor("lzma", LZMACompressorWrapper())
register_compressor("xz", XZCompressorWrapper())
register_compressor("lz4", LZ4CompressorWrapper())


###############################################################################
# Utility objects for persistence.

# For convenience, 16 bytes are used to be sure to cover all the possible
# dtypes' alignments. For reference, see:
# https://numpy.org/devdocs/dev/alignment.html
NUMPY_ARRAY_ALIGNMENT_BYTES = 16


class NumpyArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    This object is used to hack into the pickle machinery and read numpy
    array data from our custom persistence format.
    More precisely, this object is used for:
    * carrying the information of the persisted array: subclass, shape, order,
    dtype. Those ndarray metadata are used to correctly reconstruct the array
    with low level numpy functions.
    * determining if memmap is allowed on the array.
    * reading the array bytes from a file.
    * reading the array using memorymap from a file.
    * writing the array bytes to a file.

    Attributes
    ----------
    subclass: numpy.ndarray subclass
        Determine the subclass of the wrapped array.
    shape: numpy.ndarray shape
        Determine the shape of the wrapped array.
    order: {'C', 'F'}
        Determine the order of wrapped array data. 'C' is for C order, 'F' is
        for fortran order.
    dtype: numpy.ndarray dtype
        Determine the data type of the wrapped array.
    allow_mmap: bool
        Determine if memory mapping is allowed on the wrapped array.
        Default: False.
    """

    def __init__(
        self,
        subclass,
        shape,
        order,
        dtype,
        allow_mmap=False,
        numpy_array_alignment_bytes=NUMPY_ARRAY_ALIGNMENT_BYTES,
    ):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.shape = shape
        self.order = order
        self.dtype = dtype
        self.allow_mmap = allow_mmap
        # We make numpy_array_alignment_bytes an instance attribute to allow us
        # to change our mind about the default alignment and still load the old
        # pickles (with the previous alignment) correctly
        self.numpy_array_alignment_bytes = numpy_array_alignment_bytes

    def safe_get_numpy_array_alignment_bytes(self):
        # NumpyArrayWrapper instances loaded from joblib <= 1.1 pickles don't
        # have an numpy_array_alignment_bytes attribute
        return getattr(self, "numpy_array_alignment_bytes", None)

    def write_array(self, array, pickler):
        """Write array bytes to pickler file handle.

        This function is an adaptation of the numpy write_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024**2 // array.itemsize, 1)
        if array.dtype.hasobject:
            # We contain Python objects so we cannot write out the data
            # directly. Instead, we will pickle it out with version 5 of the
            # pickle protocol.
            pickle.dump(array, pickler.file_handle, protocol=5)
        else:
            numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
            if numpy_array_alignment_bytes is not None:
                current_pos = pickler.file_handle.tell()
                pos_after_padding_byte = current_pos + 1
                padding_length = numpy_array_alignment_bytes - (
                    pos_after_padding_byte % numpy_array_alignment_bytes
                )
                # A single byte is written that contains the padding length in
                # bytes
                padding_length_byte = int.to_bytes(
                    padding_length, length=1, byteorder="little"
                )
                pickler.file_handle.write(padding_length_byte)

                if padding_length != 0:
                    padding = b"\xff" * padding_length
                    pickler.file_handle.write(padding)

            for chunk in pickler.np.nditer(
                array,
                flags=["external_loop", "buffered", "zerosize_ok"],
                buffersize=buffersize,
                order=self.order,
            ):
                pickler.file_handle.write(chunk.tobytes("C"))

    def _element_count(self, np):
        """Return the number of elements described by this wrapper."""
        if len(self.shape) == 0:
            # Scalar array: a single element is stored on disk.
            return 1
        # joblib issue #859: we cast the elements of self.shape to int64 to
        # prevent a potential overflow when computing their product.
        shape_int64 = [np.int64(x) for x in self.shape]
        return int(np.multiply.reduce(shape_int64))

    def _read_and_skip_padding(self, file_handle):
        """Consume the alignment padding written before the array bytes."""
        numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()
        if numpy_array_alignment_bytes is not None:
            padding_byte = file_handle.read(1)
            padding_length = int.from_bytes(padding_byte, byteorder="little")
            if padding_length != 0:
                file_handle.read(padding_length)

    def read_array(self, unpickler, ensure_native_byte_order):
        """Read array from unpickler file handle.

        This function is an adaptation of the numpy read_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        count = self._element_count(unpickler.np)
        # Now read the actual data.
        if self.dtype.hasobject:
            # The array contained Python objects, serialized as a nested
            # pickle stream. We read it with a fresh Unpickler (so the nested
            # stream gets its own opcode stack) but route ``find_class``
            # through the *outer* unpickler instance to propagate security
            # harden behavior.
            # (see https://docs.python.org/3/library/pickle.html#pickle-restrict).
            inner_unpickler = Unpickler(unpickler.file_handle)
            inner_unpickler.find_class = unpickler.find_class
            array = inner_unpickler.load()
        else:
            self._read_and_skip_padding(unpickler.file_handle)

            # This is not a real file. We have to read it the
            # memory-intensive way.
            # crc32 module fails on reads greater than 2 ** 32 bytes,
            # breaking large reads from gzip streams. Chunk reads to
            # BUFFER_SIZE bytes to avoid issue and reduce memory overhead
            # of the read. In non-chunked case count < max_read_count, so
            # only one read is performed.
            max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, self.dtype.itemsize)

            array = unpickler.np.empty(count, dtype=self.dtype)
            for i in range(0, count, max_read_count):
                read_count = min(max_read_count, count - i)
                read_size = int(read_count * self.dtype.itemsize)
                data = _read_bytes(unpickler.file_handle, read_size, "array data")
                array[i : i + read_count] = unpickler.np.frombuffer(
                    data, dtype=self.dtype, count=read_count
                )
                del data

            if self.order == "F":
                array = array.reshape(self.shape[::-1])
                array = array.transpose()
            else:
                array = array.reshape(self.shape)

        if ensure_native_byte_order:
            # Detect byte order mismatch and swap as needed.
            array = _ensure_native_byte_order(array)

        return array

    def read_into(self, unpickler, out_array, ensure_native_byte_order):
        """Read the array bytes directly into a caller-provided destination.

        Contrary to :meth:`read_array`, this never allocates a temporary
        array of the full size of the persisted data: the stream payload is
        copied chunk by chunk, with chunks bounded by ``BUFFER_SIZE`` bytes,
        straight into the buffer of ``out_array``. The exact same object is
        returned, so the caller keeps the identity of its destination.

        The destination metadata (shape, dtype, endianness, order, capacity
        and writability) are validated before any byte is written. Any
        required native byte-order coercion happens per chunk through the
        slice assignment below, which means the destination object itself is
        never replaced by a byte-swapped copy.
        """
        if self.dtype.hasobject:
            raise ValueError(
                "out_arrays destinations are not supported for arrays with "
                "an object dtype."
            )

        self._read_and_skip_padding(unpickler.file_handle)

        np = unpickler.np
        count = self._element_count(np)
        file_handle = unpickler.file_handle

        if len(self.shape) == 0:
            # 0-d destination: assign the single element in place. Going
            # through a reshaped 1-D view would rely on reshape returning a
            # view for scalars, which is not guaranteed across numpy versions.
            data = _read_bytes(file_handle, self.dtype.itemsize, "array data")
            # The scalar assignment performs any required byte-order cast
            # without replacing the destination object.
            out_array[()] = np.frombuffer(data, dtype=self.dtype, count=1)[0]
            return out_array

        if count == 0:
            # Zero-length array: no payload byte follows, nothing to fill.
            return out_array

        # Build a 1-D, writable view of the destination that enumerates its
        # elements in the exact order used on disk (C-flat or F-flat). Both
        # ``transpose`` and ``reshape`` return views here because the
        # destination contiguity matching ``self.order`` was validated.
        if self.order == "F":
            flat = out_array.transpose().reshape(-1)
        else:
            flat = out_array.reshape(-1)

        if not np.shares_memory(flat, out_array):
            raise RuntimeError(
                "Failed to build a zero-copy 1-D view of an out_arrays "
                "destination. This is an internal joblib error."
            )

        # Chunk reads to BUFFER_SIZE bytes, including for compressed streams,
        # so that the transient work buffer never grows with the array size.
        max_read_count = BUFFER_SIZE // min(BUFFER_SIZE, self.dtype.itemsize)
        for i in range(0, count, max_read_count):
            read_count = min(max_read_count, count - i)
            read_size = int(read_count * self.dtype.itemsize)
            data = _read_bytes(file_handle, read_size, "array data")
            # The slice assignment performs the (possibly byte-swapping)
            # cast from the on-disk dtype to the destination dtype on a
            # chunk-sized buffer, without touching the destination identity.
            flat[i : i + read_count] = np.frombuffer(
                data, dtype=self.dtype, count=read_count
            )
            del data

        return out_array

    def read_mmap(self, unpickler):
        """Read an array using numpy memmap."""
        current_pos = unpickler.file_handle.tell()
        offset = current_pos
        numpy_array_alignment_bytes = self.safe_get_numpy_array_alignment_bytes()

        if numpy_array_alignment_bytes is not None:
            padding_byte = unpickler.file_handle.read(1)
            padding_length = int.from_bytes(padding_byte, byteorder="little")
            # + 1 is for the padding byte
            offset += padding_length + 1

        if unpickler.mmap_mode == "w+":
            unpickler.mmap_mode = "r+"

        marray = make_memmap(
            unpickler.filename,
            dtype=self.dtype,
            shape=self.shape,
            order=self.order,
            mode=unpickler.mmap_mode,
            offset=offset,
        )
        # update the offset so that it corresponds to the end of the read array
        unpickler.file_handle.seek(offset + marray.nbytes)

        if (
            numpy_array_alignment_bytes is None
            and current_pos % NUMPY_ARRAY_ALIGNMENT_BYTES != 0
        ):
            message = (
                f"The memmapped array {marray} loaded from the file "
                f"{unpickler.file_handle.name} is not byte aligned. "
                "This may cause segmentation faults if this memmapped array "
                "is used in some libraries like BLAS or PyTorch. "
                "To get rid of this warning, regenerate your pickle file "
                "with joblib >= 1.2.0. "
                "See https://github.com/joblib/joblib/issues/563 "
                "for more details"
            )
            warnings.warn(message)

        return marray

    def read(self, unpickler, ensure_native_byte_order):
        """Read the array corresponding to this wrapper.

        Use the unpickler to get all information to correctly read the array.

        Parameters
        ----------
        unpickler: NumpyUnpickler
        ensure_native_byte_order: bool
            If true, coerce the array to use the native endianness of the
            host system.

        Returns
        -------
        array: numpy.ndarray

        """
        # When requested, only use memmap mode if allowed.
        if unpickler.mmap_mode is not None and self.allow_mmap:
            assert not ensure_native_byte_order, (
                "Memmaps cannot be coerced to a given byte order, "
                "this code path is impossible."
            )
            array = self.read_mmap(unpickler)
        else:
            array = self.read_array(unpickler, ensure_native_byte_order)

        # Manage array subclass case
        if hasattr(array, "__array_prepare__") and self.subclass not in (
            unpickler.np.ndarray,
            unpickler.np.memmap,
        ):
            # We need to reconstruct another subclass
            new_array = _reconstruct(self.subclass, (0,), "b")
            return new_array.__array_prepare__(array)
        else:
            return array


class _OutArraysRegistry:
    """Registry of caller-provided destination arrays for :func:`load`.

    Destinations are numbered by the order in which
    :class:`NumpyArrayWrapper` instances occur in the pickle stream, starting
    at 0. Every occurrence gets its own number, including several occurrences
    originating from the same source array: joblib serializes each of them
    independently and this registry does not attempt to restore shared views
    or aliases.

    The static properties of the destinations (type, writability,
    contiguity, dtype kind and memory overlap) are validated once when the
    registry is built, before the stream is even opened. Properties that
    depend on the stream metadata (shape, dtype and endianness, order and
    capacity) are re-validated for each matched occurrence, before any byte
    is written into the destination.
    """

    def __init__(self, destinations, np):
        # Mapping of non-negative integer index -> destination ndarray.
        self._destinations = destinations
        self._np = np
        self._next_index = 0
        self._used_indices = set()

    @staticmethod
    def build(out_arrays):
        """Validate the user ``out_arrays`` argument and build a registry.

        Accepted inputs are a mapping ``{index: array}`` or a sequence where
        the position is the index (``None`` entries are ignored).
        """
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Using the out_arrays parameter of joblib.load requires "
                "numpy to be installed."
            ) from exc

        if isinstance(out_arrays, collections.abc.Mapping):
            entries = out_arrays.items()
        elif isinstance(out_arrays, (list, tuple)):
            entries = (
                (index, array)
                for index, array in enumerate(out_arrays)
                if array is not None
            )
        else:
            raise TypeError(
                "out_arrays should be a mapping of integer indices to "
                "numpy arrays, or a sequence of numpy arrays (with None "
                f"entries for unused indices), got {type(out_arrays)!r}."
            )

        destinations = {}
        for index, array in entries:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    "out_arrays indices must be non-negative integers, got "
                    f"{index!r} (type {type(index).__name__})."
                )
            if index < 0:
                raise ValueError(
                    f"out_arrays indices must be non-negative, got {index}."
                )
            if not isinstance(array, np.ndarray):
                raise TypeError(
                    f"out_arrays[{index}] must be a numpy.ndarray or "
                    f"numpy.memmap instance, got {type(array)!r}."
                )
            if not array.flags.writeable:
                raise ValueError(
                    f"out_arrays[{index}] is read-only: destinations must "
                    "be writable arrays."
                )
            if not (array.flags.c_contiguous or array.flags.f_contiguous):
                raise ValueError(
                    f"out_arrays[{index}] is neither C-contiguous nor "
                    "F-contiguous: only contiguous destinations are "
                    "supported."
                )
            if array.dtype.hasobject:
                raise ValueError(
                    f"out_arrays[{index}] has an object dtype, which is not supported."
                )
            destinations[index] = array

        # Reject destinations sharing memory: filling them independently
        # could silently corrupt one destination through another one.
        indices = sorted(destinations)
        for pos, index in enumerate(indices):
            for other_index in indices[pos + 1 :]:
                if np.shares_memory(destinations[index], destinations[other_index]):
                    raise ValueError(
                        f"out_arrays[{index}] and out_arrays[{other_index}] "
                        "share memory: overlapping destinations are not "
                        "supported."
                    )

        return _OutArraysRegistry(destinations, np)

    def begin_wrapper(self, wrapper, ensure_native_byte_order):
        """Consume one wrapper occurrence and return its destination or None.

        Raises a ``ValueError`` if a destination was provided for this
        occurrence but is incompatible with the stream metadata. The stream
        position is not moved when such an error is raised.
        """
        index = self._next_index
        self._next_index += 1

        out_array = self._destinations.get(index)
        if out_array is None:
            return None

        self._validate_for_wrapper(index, out_array, wrapper, ensure_native_byte_order)
        self._used_indices.add(index)
        return out_array

    def unused_indices(self):
        """Return the sorted indices that never matched a stream array."""
        return sorted(set(self._destinations) - self._used_indices)

    @staticmethod
    def _expected_dtype(wrapper, ensure_native_byte_order):
        # When native byte order is enforced, the destination is expected to
        # carry the native-endian dtype and the per-chunk slice assignment in
        # ``read_into`` performs the byte swap without replacing the array.
        if ensure_native_byte_order:
            return wrapper.dtype.newbyteorder("=")
        return wrapper.dtype

    def _validate_for_wrapper(
        self, index, out_array, wrapper, ensure_native_byte_order
    ):
        if wrapper.dtype.hasobject:
            raise ValueError(
                f"out_arrays[{index}] targets an array with an object "
                "dtype, which is not supported by out_arrays."
            )

        # The on-disk element order must match the destination layout, so
        # that the flat payload can be streamed in without transposing.
        if wrapper.order == "C" and not out_array.flags.c_contiguous:
            raise ValueError(
                f"out_arrays[{index}] is not C-contiguous but the stream "
                "array is stored in C order."
            )
        if wrapper.order == "F" and not out_array.flags.f_contiguous:
            raise ValueError(
                f"out_arrays[{index}] is not F-contiguous but the stream "
                "array is stored in Fortran order."
            )

        if tuple(out_array.shape) != tuple(wrapper.shape):
            raise ValueError(
                f"out_arrays[{index}] shape {out_array.shape} does not "
                f"match the stream array shape {tuple(wrapper.shape)}."
            )

        expected_dtype = self._expected_dtype(wrapper, ensure_native_byte_order)
        if out_array.dtype != expected_dtype:
            policy = "native byte order" if ensure_native_byte_order else "stream"
            raise ValueError(
                f"out_arrays[{index}] dtype {out_array.dtype!r} does not "
                f"match the {policy} dtype {expected_dtype!r} of the stream "
                "array (endianness included)."
            )

        element_count = wrapper._element_count(self._np)
        required_bytes = element_count * wrapper.dtype.itemsize
        if out_array.nbytes < required_bytes:
            raise ValueError(
                f"out_arrays[{index}] capacity ({out_array.nbytes} bytes) "
                f"is too small for the stream array ({required_bytes} "
                "bytes)."
            )


###############################################################################
# Pickler classes


class NumpyPickler(Pickler):
    """A pickler to persist big data efficiently.

    The main features of this object are:
    * persistence of numpy arrays in a single file.
    * optional compression with a special care on avoiding memory copies.

    Attributes
    ----------
    fp: file
        File object handle used for serializing the input object.
    protocol: int, optional
        Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL.
    """

    dispatch = Pickler.dispatch.copy()

    def __init__(self, fp, protocol=None):
        self.file_handle = fp
        self.buffered = isinstance(self.file_handle, BinaryZlibFile)

        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        if protocol is None:
            protocol = pickle.DEFAULT_PROTOCOL

        Pickler.__init__(self, self.file_handle, protocol=protocol)
        # delayed import of numpy, to avoid tight coupling
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _create_array_wrapper(self, array):
        """Create and returns a numpy array wrapper from a numpy array."""
        order = (
            "F" if (array.flags.f_contiguous and not array.flags.c_contiguous) else "C"
        )
        allow_mmap = not self.buffered and not array.dtype.hasobject

        kwargs = {}
        try:
            self.file_handle.tell()
        except io.UnsupportedOperation:
            kwargs = {"numpy_array_alignment_bytes": None}

        wrapper = NumpyArrayWrapper(
            type(array),
            array.shape,
            order,
            array.dtype,
            allow_mmap=allow_mmap,
            **kwargs,
        )

        return wrapper

    def save(self, obj):
        """Subclass the Pickler `save` method.

        This is a total abuse of the Pickler class in order to use the numpy
        persistence function `save` instead of the default pickle
        implementation. The numpy array is replaced by a custom wrapper in the
        pickle persistence stack and the serialized array is written right
        after in the file. Warning: the file produced does not follow the
        pickle format. As such it can not be read with `pickle.load`.
        """
        if self.np is not None and type(obj) in (
            self.np.ndarray,
            self.np.matrix,
            self.np.memmap,
        ):
            if type(obj) is self.np.memmap:
                # Pickling doesn't work with memmapped arrays
                obj = self.np.asanyarray(obj)

            # The array wrapper is pickled instead of the real array.
            wrapper = self._create_array_wrapper(obj)
            Pickler.save(self, wrapper)

            # A framer was introduced with pickle protocol 4 and we want to
            # ensure the wrapper object is written before the numpy array
            # buffer in the pickle file.
            # See https://www.python.org/dev/peps/pep-3154/#framing to get
            # more information on the framer behavior.
            if self.proto >= 4:
                self.framer.commit_frame(force=True)

            # And then array bytes are written right after the wrapper.
            wrapper.write_array(obj, self)
            return

        return Pickler.save(self, obj)


class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.

    Attributes
    ----------
    mmap_mode: str
        The memorymap mode to use for reading numpy arrays.
    file_handle: file_like
        File object to unpickle from.
    ensure_native_byte_order: bool
        If True, coerce the array to use the native endianness of the
        host system.
    filename: str
        Name of the file to unpickle from. It should correspond to file_handle.
        This parameter is required when using mmap_mode.
    np: module
        Reference to numpy module if numpy is installed else None.

    """

    dispatch = Unpickler.dispatch.copy()

    def __init__(
        self,
        filename,
        file_handle,
        ensure_native_byte_order,
        mmap_mode=None,
        out_arrays=None,
    ):
        # The next line is for backward compatibility with pickle generated
        # with joblib versions less than 0.10.
        self._dirname = os.path.dirname(filename)

        self.mmap_mode = mmap_mode
        self.file_handle = file_handle
        # filename is required for numpy mmap mode.
        self.filename = filename
        self.compat_mode = False
        self.ensure_native_byte_order = ensure_native_byte_order
        # _OutArraysRegistry holding the caller-provided destination arrays,
        # numbered by NumpyArrayWrapper occurrence order. None disables the
        # direct-write path entirely.
        self.out_arrays = out_arrays
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def load_build(self):
        """Called to set the state of a newly created object.

        We capture it to replace our place-holder objects, NDArrayWrapper or
        NumpyArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        NDArrayWrapper is used for backward compatibility with joblib <= 0.9.
        """
        Unpickler.load_build(self)

        # For backward compatibility, we support NDArrayWrapper objects.
        if isinstance(self.stack[-1], (NDArrayWrapper, NumpyArrayWrapper)):
            if self.np is None:
                raise ImportError(
                    "Trying to unpickle an ndarray, but numpy didn't import correctly"
                )
            array_wrapper = self.stack.pop()
            # If any NDArrayWrapper is found, we switch to compatibility mode,
            # this will be used to raise a DeprecationWarning to the user at
            # the end of the unpickling.
            try:
                if isinstance(array_wrapper, NDArrayWrapper):
                    self.compat_mode = True
                    _array_payload = array_wrapper.read(self)
                elif self.out_arrays is not None:
                    # Numbering counts actual NumpyArrayWrapper occurrences.
                    # A matched destination is filled in place, preserving
                    # its object identity; unmatched arrays fall back to the
                    # regular allocation-based loading path.
                    out_array = self.out_arrays.begin_wrapper(
                        array_wrapper, self.ensure_native_byte_order
                    )
                    if out_array is not None:
                        _array_payload = array_wrapper.read_into(
                            self, out_array, self.ensure_native_byte_order
                        )
                    else:
                        _array_payload = array_wrapper.read(
                            self, self.ensure_native_byte_order
                        )
                else:
                    _array_payload = array_wrapper.read(
                        self, self.ensure_native_byte_order
                    )
            except ValueError as exc:
                if str(exc) == "Array can't be memory-mapped: Python objects in dtype.":
                    raise readPython2Error(exc)
                raise

            self.stack.append(_array_payload)

    # Be careful to register our new method.
    dispatch[pickle.BUILD[0]] = load_build


###############################################################################
# Utility functions


def dump(value, filename, compress=0, protocol=None):
    """Persist an arbitrary Python object into one file.

    Read more in the :ref:`User Guide <persistence>`.

    Parameters
    ----------
    value: any Python object
        The object to store to disk.
    filename: str, os.PathLike, or file object.
        The file object or path of the file in which it is to be stored.
        The compression method corresponding to one of the supported filename
        extensions ('.z', '.gz', '.bz2', '.xz' or '.lzma') will be used
        automatically.
    compress: int from 0 to 9 or bool or 2-tuple, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher value means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
        If compress is True, the compression level used is 3.
        If compress is a 2-tuple, the first element must correspond to a string
        between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
        'xz'), the second element must be an integer from 0 to 9, corresponding
        to the compression level.
    protocol: int, optional
        Pickle protocol, see pickle.dump documentation for more details.

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    See Also
    --------
    joblib.load : corresponding loader

    Notes
    -----
    Memmapping on load cannot be used for compressed files. Thus
    using compression can significantly slow down loading. In
    addition, compressed files take up extra memory during
    dump and load.

    """

    if isinstance(filename, os.PathLike):
        filename = os.fspath(filename)

    is_filename = isinstance(filename, str)
    is_fileobj = hasattr(filename, "write")

    compress_method = "zlib"  # zlib is the default compression method.
    if compress is True:
        # By default, if compress is enabled, we want the default compress
        # level of the compressor.
        compress_level = None
    elif isinstance(compress, tuple):
        # a 2-tuple was set in compress
        if len(compress) != 2:
            raise ValueError(
                "Compress argument tuple should contain exactly 2 elements: "
                "(compress method, compress level), you passed {}".format(compress)
            )
        compress_method, compress_level = compress
    elif isinstance(compress, str):
        compress_method = compress
        compress_level = None  # Use default compress level
        compress = (compress_method, compress_level)
    else:
        compress_level = compress

    if compress_method == "lz4" and lz4 is None:
        raise ValueError(LZ4_NOT_INSTALLED_ERROR)

    if (
        compress_level is not None
        and compress_level is not False
        and compress_level not in range(10)
    ):
        # Raising an error if a non valid compress level is given.
        raise ValueError(
            'Non valid compress level given: "{}". Possible values are {}.'.format(
                compress_level, list(range(10))
            )
        )

    if compress_method not in _COMPRESSORS:
        # Raising an error if an unsupported compression method is given.
        raise ValueError(
            'Non valid compression method given: "{}". Possible values are {}.'.format(
                compress_method, _COMPRESSORS
            )
        )

    if not is_filename and not is_fileobj:
        # People keep inverting arguments, and the resulting error is
        # incomprehensible
        raise ValueError(
            "Second argument should be a filename or a file-like object, "
            "%s (type %s) was given." % (filename, type(filename))
        )

    if is_filename and not isinstance(compress, tuple):
        # In case no explicit compression was requested using both compression
        # method and level in a tuple and the filename has an explicit
        # extension, we select the corresponding compressor.

        # unset the variable to be sure no compression level is set afterwards.
        compress_method = None
        for name, compressor in _COMPRESSORS.items():
            if filename.endswith(compressor.extension):
                compress_method = name

        if compress_method in _COMPRESSORS and compress_level == 0:
            # we choose the default compress_level in case it was not given
            # as an argument (using compress).
            compress_level = None

    if compress_level != 0:
        with _write_fileobject(
            filename, compress=(compress_method, compress_level)
        ) as f:
            NumpyPickler(f, protocol=protocol).dump(value)
    elif is_filename:
        with open(filename, "wb") as f:
            NumpyPickler(f, protocol=protocol).dump(value)
    else:
        NumpyPickler(filename, protocol=protocol).dump(value)

    # If the target container is a file object, nothing is returned.
    if is_fileobj:
        return

    # For compatibility, the list of created filenames (e.g with one element
    # after 0.10.0) is returned by default.
    return [filename]


def _unpickle(
    fobj,
    ensure_native_byte_order,
    filename="",
    mmap_mode=None,
    out_arrays=None,
):
    """Internal unpickling function."""
    # We are careful to open the file handle early and keep it open to
    # avoid race-conditions on renames.
    # That said, if data is stored in companion files, which can be
    # the case with the old persistence format, moving the directory
    # will create a race when joblib tries to access the companion
    # files.
    unpickler = NumpyUnpickler(
        filename,
        fobj,
        ensure_native_byte_order,
        mmap_mode=mmap_mode,
        out_arrays=out_arrays,
    )
    obj = None
    try:
        obj = unpickler.load()
        if unpickler.compat_mode:
            warnings.warn(
                "The file '%s' has been generated with a "
                "joblib version less than 0.10. "
                "Please regenerate this pickle file." % filename,
                DeprecationWarning,
                stacklevel=3,
            )
    except UnicodeDecodeError as exc:
        raise readPython2Error(exc)
    return obj


def load_temporary_memmap(filename, mmap_mode, unlink_on_gc_collect):
    from ._memmapping_reducer import JOBLIB_MMAPS, add_maybe_unlink_finalizer

    with open(filename, "rb") as f:
        with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
            fobj,
            validated_mmap_mode,
        ):
            # Memmap are used for interprocess communication, which should
            # keep the objects untouched. We pass `ensure_native_byte_order=False`
            # to remain consistent with the loading behavior of non-memmaped arrays
            # in workers, where the byte order is preserved.
            # Note that we do not implement endianness change for memmaps, as this
            # would result in inconsistent behavior.
            obj = _unpickle(
                fobj,
                ensure_native_byte_order=False,
                filename=filename,
                mmap_mode=validated_mmap_mode,
            )

    JOBLIB_MMAPS.add(obj.filename)
    if unlink_on_gc_collect:
        add_maybe_unlink_finalizer(obj)
    return obj


def load(
    filename,
    mmap_mode=None,
    ensure_native_byte_order="auto",
    out_arrays=None,
):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Read more in the :ref:`User Guide <persistence>`.

    WARNING: joblib.load relies on the pickle module and can therefore
    execute arbitrary Python code. It should therefore never be used
    to load files from untrusted sources.

    Parameters
    ----------
    filename: str, os.PathLike, or file object.
        The file object or path of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has no effect for compressed files. Note that in this
        case the reconstructed object might no longer match exactly
        the originally pickled object.
    ensure_native_byte_order: bool, or 'auto', default=='auto'
        If True, ensures that the byte order of the loaded arrays matches the
        native byte ordering (or _endianness_) of the host system. This is not
        compatible with memory-mapped arrays and using non-null `mmap_mode`
        parameter at the same time will raise an error. The default 'auto'
        parameter is equivalent to True if `mmap_mode` is None, else False.
    out_arrays: mapping or sequence of numpy.ndarray, optional
        Pre-allocated writable arrays (including ``numpy.memmap`` instances)
        into which the persisted numpy arrays are filled directly, instead of
        allocating a new array of the same size and copying into it. Keys are
        integers numbering the arrays by the order in which the corresponding
        ``NumpyArrayWrapper`` objects occur in the stream, starting at 0. A
        mapping (e.g. ``{0: array_0, 2: array_2}``) only overrides the given
        indices; a sequence is also accepted and ``None`` entries are ignored.
        Arrays whose index is not provided keep the regular loading behavior.

        Each destination must be a writable, C- or F-contiguous, non-object
        ``numpy.ndarray``. Before a destination is written, its shape, dtype
        (including endianness), on-disk order compatibility and byte capacity
        are validated against the stream metadata; an incompatible, read-only
        or overlapping destination raises an error. A stream stored in C
        order requires a C-contiguous destination, and likewise for Fortran
        order. Scalar and zero-length arrays are supported.

        When ``ensure_native_byte_order`` is True, the destination dtype must
        already use the native byte order: the conversion is applied chunk by
        chunk while filling the destination, the destination object itself is
        never replaced by a byte-swapped copy. When it is False, the
        destination dtype must match the raw on-disk dtype exactly.

        ``mmap_mode`` only applies to arrays that are not listed in
        ``out_arrays``: listed arrays are always filled directly from the
        stream, including in compressed files. If a destination is itself a
        caller-created ``numpy.memmap``, joblib never closes or deletes its
        backing file.

        Filling is not transactional: if the stream is truncated or corrupted
        after a destination has started being filled, that destination is left
        partially rewritten and the error is propagated, without ever returning
        a partially reconstructed object. Indices provided in ``out_arrays``
        that no stream array matches are reported through a warning once the
        whole stream has been read.

    Returns
    -------
    result: any Python object
        The object stored in the file. Arrays matched through ``out_arrays``
        are the exact destination objects passed by the caller (same identity
        and memory).

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump. If the mmap_mode argument is given, it is passed to np.load and
    arrays are loaded as memmaps. As a consequence, the reconstructed
    object might not match the original pickled object. Note that if the
    file was saved with compression, the arrays cannot be memmapped.
    """
    if ensure_native_byte_order == "auto":
        ensure_native_byte_order = mmap_mode is None

    if ensure_native_byte_order and mmap_mode is not None:
        raise ValueError(
            "Native byte ordering can only be enforced if 'mmap_mode' parameter "
            f"is set to None, but got 'mmap_mode={mmap_mode}' instead."
        )

    out_arrays_registry = None
    if out_arrays is not None:
        # Static validation (type, writability, contiguity, overlap) happens
        # before the stream is opened, so an invalid argument never causes a
        # partial write.
        out_arrays_registry = _OutArraysRegistry.build(out_arrays)

    if isinstance(filename, os.PathLike):
        filename = os.fspath(filename)

    if hasattr(filename, "read"):
        fobj = filename
        filename = getattr(fobj, "name", "")
        with _validate_fileobject_and_memmap(fobj, filename, mmap_mode) as (fobj, _):
            obj = _unpickle(
                fobj,
                ensure_native_byte_order=ensure_native_byte_order,
                out_arrays=out_arrays_registry,
            )
    else:
        with open(filename, "rb") as f:
            with _validate_fileobject_and_memmap(f, filename, mmap_mode) as (
                fobj,
                validated_mmap_mode,
            ):
                if isinstance(fobj, str):
                    # if the returned file object is a string, this means we
                    # try to load a pickle file generated with an version of
                    # Joblib so we load it with joblib compatibility function.
                    if out_arrays_registry is not None:
                        raise ValueError(
                            "The out_arrays parameter is not supported for "
                            "pickle files generated with joblib < 0.10."
                        )
                    return load_compatibility(fobj)

                # A memory-mapped array has to be mapped with the endianness
                # it has been written with. Other arrays are coerced to the
                # native endianness of the host system.
                obj = _unpickle(
                    fobj,
                    ensure_native_byte_order=ensure_native_byte_order,
                    filename=filename,
                    mmap_mode=validated_mmap_mode,
                    out_arrays=out_arrays_registry,
                )

    if out_arrays_registry is not None:
        unused_indices = out_arrays_registry.unused_indices()
        if unused_indices:
            warnings.warn(
                "The following out_arrays indices did not match any numpy "
                f"array in the stream and were not used: {unused_indices}.",
                UserWarning,
                stacklevel=2,
            )

    return obj
