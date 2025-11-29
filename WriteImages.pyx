# Thread-local storage for per-thread encoder reuse
import threading
_thread_state = threading.local()

import os
import numpy as np
from multiprocessing import shared_memory

from libc.stdio cimport FILE, fopen, fclose
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

from WriteImages cimport (
    spng_ctx_new, spng_ctx_free, spng_set_option, spng_encode_image,
    SpngCtxWrapper, _unwrap_ctx, _spng_encode,
)

# ----------------------------------------------------------
# Constants (must match libspng)
# ----------------------------------------------------------
cdef int SPNG_FMT_RGB8 = 4
cdef int SPNG_ENCODE_FINALIZE = 2 # Finalize PNG after encoding image




# ----------------------------------------------------------
# SpngCtxWrapper implementation
# ----------------------------------------------------------
cdef class SpngCtxWrapper:

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            spng_ctx_free(self.ptr)
            self.ptr = NULL


cdef spng_ctx* _unwrap_ctx(SpngCtxWrapper wrap) except NULL nogil:
    return wrap.ptr

# ----------------------------------------------------------
# TLS-based lazy context initialization
# ----------------------------------------------------------
def _get_thread_ctx_py(int compression_level):
    """
    Returns or creates a thread-local SpngCtxWrapper instance.
    """
    cdef SpngCtxWrapper wrap = getattr(_thread_state, "ctx", None)

    if wrap is not None:
        return wrap

    wrap = SpngCtxWrapper()
    wrap.ptr = spng_ctx_new(0)
    if wrap.ptr == NULL:
        raise MemoryError("spng_ctx_new failed")

    # configure compression (GIL held)
    if spng_set_option(wrap.ptr, 3, compression_level) != 0:
        spng_ctx_free(wrap.ptr)
        wrap.ptr = NULL
        raise RuntimeError("spng_set_option failed")

    _thread_state.ctx = wrap
    return wrap


# ----------------------------------------------------------
# GIL-free pure C PNG encoder helper
# ----------------------------------------------------------
cdef int _spng_encode(
    spng_ctx* ctx,
    uint8_t* img_ptr,
    size_t img_len,
    int fmt,
    int flags
) except -1 nogil:
    return spng_encode_image(ctx, <const void*>img_ptr, img_len, fmt, flags)


# ----------------------------------------------------------
# PNG encode wrapper (manages FILE* but uses ctx under GIL)
# ----------------------------------------------------------
def encode_png_spng(bytes filename,
                    uint8_t* img_ptr,
                    int width, int height, int stride,
                    int compression_level):
    """
    Calls libspng to encode one frame via thread-local ctx.
    """
    cdef FILE* fp
    cdef SpngCtxWrapper wrap
    cdef spng_ctx* ctx
    cdef int ret

    fp = fopen(filename, b"wb")
    if fp == NULL:
        raise IOError(f"fopen failed for {filename!r}")

    try:
        wrap = _get_thread_ctx_py(compression_level)
    except Exception:
        fclose(fp)
        raise

    ctx = _unwrap_ctx(wrap)
    if ctx == NULL:
        fclose(fp)
        raise RuntimeError("Invalid spng_ctx pointer")

    with nogil:
        ret = _spng_encode(ctx, img_ptr, stride * height, SPNG_FMT_RGB8, SPNG_ENCODE_FINALIZE)

    fclose(fp)

    if ret != 0:
        raise RuntimeError(f"spng_encode_image failed with code {ret}")


# ----------------------------------------------------------
# Entry point used by multiprocessing.Process / apply_async
# ----------------------------------------------------------
def write_images(bytes shm_name,
                 bytes desc_name,
                 int frames_total,
                 int width,
                 int height,
                 bytes output_path,
                 int compression_level):
    """
    Worker entrypoint: maps shared memory,
    reads descriptor array and writes PNGs.
    """
    cdef int i
    cdef uint8_t* frame_ptr

    # Used to build filenames
    cdef int img_bytes, frame_count, bracket_index

    cdef str shm_str = shm_name.decode()
    cdef str desc_str = desc_name.decode()
    cdef str base = output_path.decode()

    # ----------------------------------------------------------
    # Attach image shared memory buffer
    # ----------------------------------------------------------
    shm = shared_memory.SharedMemory(name=shm_str)
    buf = np.ndarray((frames_total, height, width, 3),
                     dtype=np.uint8,
                     buffer=shm.buf)

    # ----------------------------------------------------------
    # Attach descriptor shared memory buffer
    # Layout guaranteed by DigitizeVideo.pyx:
    #   desc_arr[i,0] = img_bytes (uint32)
    #   desc_arr[i,1] = frame_count (uint32)
    #   desc_arr[i,2] = bracket_index (uint32)
    # ----------------------------------------------------------
    desc_shm = shared_memory.SharedMemory(name=desc_str)
    desc_arr = np.ndarray((frames_total, 3),
                          dtype=np.uint32,
                          buffer=desc_shm.buf)

    # ----------------------------------------------------------
    # Encode each frame as PNG using the descriptor metadata
    # ----------------------------------------------------------
    for i in range(frames_total):
        # Extract descriptor metadata
        img_bytes     = int(desc_arr[i, 0])
        frame_count   = int(desc_arr[i, 1])
        bracket_index = int(desc_arr[i, 2])

        # Build descriptive filename
        filename = os.path.join(
            base,
            f"f{frame_count:05d}_b{bracket_index}_s{img_bytes}.png"
        )

        frame_ptr = <uint8_t*>buf[i].ctypes.data

        encode_png_spng(filename.encode(),
                        frame_ptr,
                        width, height,
                        width * 3,
                        compression_level)

    shm.close()
    desc_shm.close()
