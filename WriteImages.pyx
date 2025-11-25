# distutils: language = c
# cython: boundscheck=False, wraparound=False, initializedcheck=False
# cython: nonecheck=False, cdivision=True, infer_types=True

from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fopen, fclose

cimport numpy as cnp
import numpy as np


# ============================================================
# C libspng API
# ============================================================

cdef extern from "spng.h":

    ctypedef struct spng_ctx:
        pass

    cdef struct spng_ihdr:
        uint32_t width
        uint32_t height
        uint8_t  bit_depth
        uint8_t  color_type
        uint8_t  compression_method
        uint8_t  filter_method
        uint8_t  interlace_method

    # error → message
    const char *spng_strerror(int err)

    # context
    spng_ctx *spng_ctx_new(int flags)
    void spng_ctx_free(spng_ctx *ctx)

    # set IHDR
    int spng_set_ihdr(spng_ctx *ctx, spng_ihdr *ihdr)

    # set output file
    int spng_set_png_file(spng_ctx *ctx, FILE *fp)

    # encode from raw RGB buffer
    int spng_encode_image(
        spng_ctx *ctx,
        const void *src,
        size_t src_len,
        int fmt,
        int flags
    )

    # encoder options
    int spng_set_option(spng_ctx *ctx, int option, int value)

    # constants
    int SPNG_CTX_ENCODER
    int SPNG_FMT_RAW
    int SPNG_ENCODE_FINALIZE

    int SPNG_ENCODE_TO_FILE

    # pixel formats
    int SPNG_COLOR_TYPE_TRUECOLOR


# ============================================================
# Python-level wrapper
# ============================================================

cdef inline void _raise_spng_error(int err):
    """Raise Python RuntimeError with libspng message."""
    if err != 0:
        raise RuntimeError(f"libspng error {err}: {spng_strerror(err).decode()}")


cpdef write_images(
    list frames,
    list filenames,
    int compression_level = 6
):
    """
    Write a batch of RGB images (frames) to disk using libspng.

    Parameters
    ----------
    frames : list of numpy arrays (uint8, H×W×3)
    filenames : list of output PNG filenames
    compression_level : 0–12 (recommended: 3–9)

    The function is intentionally Python-level only at the outer loop;
    libspng handles the heavy work inside C code.
    """

    cdef:
        int i, height, width, err
        uint8_t *buf_ptr
        size_t buf_len
        cnp.uint8_t[:, :, :] view
        spng_ctx *ctx
        spng_ihdr ihdr
        FILE *fp

    if len(frames) != len(filenames):
        raise ValueError("frames and filenames lists must be of equal length")

    for i in range(len(frames)):
        frame = frames[i]
        fn = filenames[i]

        # ------ Validate & obtain memoryview ------
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be H×W×3 RGB uint8")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)

        view = frame

        height = view.shape[0]
        width = view.shape[1]

        buf_ptr = &view[0, 0, 0]
        buf_len = <size_t>(width * height * 3)

        # ====================================================
        # Create encoder
        # ====================================================
        ctx = spng_ctx_new(SPNG_CTX_ENCODER)
        if ctx == NULL:
            raise MemoryError("Failed to create spng_ctx")

        # ====================================================
        # Build IHDR chunk
        # ====================================================
        ihdr.width = width
        ihdr.height = height
        ihdr.bit_depth = 8
        ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR
        ihdr.compression_method = 0
        ihdr.filter_method = 0
        ihdr.interlace_method = 0

        err = spng_set_ihdr(ctx, &ihdr)
        _raise_spng_error(err)

        # ====================================================
        # Configure compression
        # ====================================================
        # libspng option 1 = compression level
        err = spng_set_option(ctx, 1, compression_level)
        _raise_spng_error(err)

        # ====================================================
        # Open output file
        # ====================================================
        fp = fopen(fn.encode("utf-8"), "wb")
        if fp == NULL:
            spng_ctx_free(ctx)
            raise IOError(f"Could not open {fn}")

        err = spng_set_png_file(ctx, fp)
        if err != 0:
            fclose(fp)
            spng_ctx_free(ctx)
            _raise_spng_error(err)

        # ====================================================
        # Encode PNG
        # ====================================================
        err = spng_encode_image(
            ctx,
            <const void *>buf_ptr,
            buf_len,
            SPNG_FMT_RAW,
            SPNG_ENCODE_FINALIZE
        )

        # cleanup before raising
        fclose(fp)
        if err != 0:
            spng_ctx_free(ctx)
            _raise_spng_error(err)

        spng_ctx_free(ctx)

    return None
