from libc.stddef cimport size_t
from libc.stdint cimport uint8_t

cdef extern from "spng.h":
    cdef struct spng_ctx

    spng_ctx* spng_ctx_new(int) nogil
    void      spng_ctx_free(spng_ctx*) nogil
    int       spng_set_option(spng_ctx*, int, int) nogil

    int spng_encode_image(
        spng_ctx*,
        const void* buf,
        size_t len,
        int fmt,
        int flags
    ) nogil


cdef class SpngCtxWrapper:
    cdef spng_ctx* ptr


# unwrap helper
cdef spng_ctx* _unwrap_ctx(SpngCtxWrapper wrap) except NULL nogil


# pure nogil C-level encode wrapper
cdef int _spng_encode(
    spng_ctx* ctx,
    uint8_t* img_ptr,
    size_t img_len,
    int fmt,
    int flags
) except -1 nogil
