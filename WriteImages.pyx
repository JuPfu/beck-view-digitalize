# Fixed-width integer types
cdef extern from "stdint.h":
    ctypedef unsigned char  uint8_t
    ctypedef unsigned int   uint32_t

# FILE* type
cdef extern from "stdio.h":
    ctypedef struct FILE:
        pass

# libspng headers
cdef extern from "spng.h":

    # --- struct definitions ---

    ctypedef struct spng_ihdr:
        uint32_t width
        uint32_t height
        uint8_t  bit_depth
        uint8_t  color_type
        uint8_t  compression_method
        uint8_t  filter_method
        uint8_t  interlace_method

    ctypedef struct spng_ctx:
        pass

    # --- constants / enums ---

    int SPNG_ENCODE_FINALIZE
    int SPNG_COLOR_TYPE_TRUECOLOR
    int SPNG_FMT_RAW
    int SPNG_FILTER_CHOICE_ALL

    # Option codes (compression tuning)
    int SPNG_IMG_COMPRESSION_LEVEL
    int SPNG_IMG_MEM_LEVEL
    int SPNG_IMG_WINDOW_BITS

    # --- functions ---

    spng_ctx *spng_ctx_new(int flags)
    void      spng_ctx_free(spng_ctx *ctx)

    int spng_set_option(spng_ctx *ctx, int option, int value)
    int spng_set_ihdr(spng_ctx *ctx, spng_ihdr *ihdr)

    int spng_encode_image(
        spng_ctx *ctx,
        const void *img,
        size_t len,
        int fmt,
        int flags
    )

    int spng_set_png_file(spng_ctx *ctx, FILE *fp)

    const char *spng_strerror(int error)
