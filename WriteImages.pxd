from libc.stdio cimport FILE

cdef extern from "spng.h":
    cdef struct spng_ctx
    spng_ctx* spng_ctx_new(int)
    void spng_ctx_free(spng_ctx*)
    int spng_set_option(spng_ctx*, int, int)
    int spng_encode_image(spng_ctx*, const void* buf, size_t len, int fmt, int flags)
    int SPNG_FMT_RGB8
    int SPNG_ENCODE_FINALIZE

cdef class SpngCtxWrapper:
    cdef spng_ctx* ptr

# Nogil-safe C helper to unwrap ctx
cdef spng_ctx* _unwrap_ctx(SpngCtxWrapper wrap) nogil except NULL
