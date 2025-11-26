# Thread-local storage
import threading
_thread_state = threading.local()

from libc.stdio cimport FILE, fopen, fclose
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
import numpy as np
import os
from multiprocessing import shared_memory


# ----------------------------------------------------
# Wrapper class
# ----------------------------------------------------
cdef class SpngCtxWrapper:

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            spng_ctx_free(self.ptr)
            self.ptr = NULL

# ----------------------------------------------------
# Nogil-safe C helper
# ----------------------------------------------------
cdef spng_ctx* _unwrap_ctx(SpngCtxWrapper wrap) nogil except NULL:
    if wrap is not None:
        return wrap.ptr
    return NULL

# ----------------------------------------------------
# Thread-local context management
# ----------------------------------------------------
def _get_thread_ctx_py(int compression_level):
    cdef SpngCtxWrapper wrap
    wrap = getattr(_thread_state, "ctx", None)

    if wrap is not None:
        return wrap

    # create wrapper and underlying ctx
    wrap = SpngCtxWrapper()
    wrap.ptr = spng_ctx_new(0)
    if wrap.ptr == NULL:
        raise MemoryError("spng_ctx_new failed")

    # Set compression while holding GIL
    spng_set_option(wrap.ptr, 3, compression_level)

    # Store in TLS
    _thread_state.ctx = wrap
    return wrap

# ----------------------------------------------------
# Python-visible PNG encoder
# ----------------------------------------------------
def encode_png_spng(bytes filename,
                    uint8_t* img_ptr,
                    int width, int height, int stride,
                    int compression_level):
    """
    Encode single PNG using thread-local spng_ctx
    """
    cdef SpngCtxWrapper wrap
    cdef spng_ctx* ctx
    cdef FILE* fp

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

    # encoding call
    cdef int rc = spng_encode_image(ctx, img_ptr, <size_t>(stride * height), SPNG_FMT_RGB8, SPNG_ENCODE_FINALIZE)
    if rc != 0:
        raise RuntimeError(f"spng_encode_image failed with code {rc}")

    fclose(fp)

# ----------------------------------------------------
# Python-visible entry point for process pool
# ----------------------------------------------------
def write_images(bytes shm_name,
                 bytes desc_name,
                 int frames_total,
                 int width,
                 int height,
                 bytes output_path,
                 int compression_level):
    """
    Maps shared memory and writes frames as PNG using thread-local ctx
    """
    cdef SpngCtxWrapper wrap
    cdef spng_ctx* ctx
    cdef uint8_t* frame_ptr
    cdef int i
    import numpy as np

    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name.decode())
    buf = np.ndarray((frames_total, height, width, 3), dtype=np.uint8, buffer=shm.buf)

    for i in range(frames_total):
        filename = os.path.join(output_path.decode(), f"{desc_name.decode()}_{i:05d}.png")
        frame_ptr = <uint8_t*>buf[i].ctypes.data
        encode_png_spng(filename.encode(), frame_ptr, width, height, width*3, compression_level)

    shm.close()
    shm.unlink()
