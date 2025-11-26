# WriteImages.pxd
# Minimal declarations needed by WriteImages.pyx

from libc.stdint cimport uint32_t, uint8_t
from libc.stddef cimport size_t
from libc.stdio cimport FILE

cdef extern from "spng.h":
    # Opaque context
    ctypedef struct spng_ctx:
        pass

    # Correct, fully declared struct spng_ihdr
    cdef struct spng_ihdr:
        uint32_t width
        uint32_t height
        uint8_t bit_depth
        uint8_t color_type
        uint8_t compression_method
        uint8_t filter_method
        uint8_t interlace_method

    # Core API
    spng_ctx *spng_ctx_new(int flags)
    void spng_ctx_free(spng_ctx *ctx)

    int spng_set_png_file(spng_ctx *ctx, FILE *fp)
    int spng_set_option(spng_ctx *ctx, int option, int value)
    int spng_set_ihdr(spng_ctx *ctx, spng_ihdr *ihdr)

    int spng_encode_image(spng_ctx *ctx, const void *img, size_t len,
                          int fmt, int flags)

    const char *spng_strerror(int err)
    const char *spng_version_string()
