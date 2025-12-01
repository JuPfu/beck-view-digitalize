# WriteImages.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import os
import sys
import logging

cimport numpy as np
import numpy as np

from multiprocessing import shared_memory

from libc.stdio cimport FILE, fopen, fclose
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

from WriteImages cimport (
    png_structp, png_infop,
    png_create_write_struct, png_create_info_struct,
    png_destroy_write_struct,
    png_init_io, png_set_IHDR,
    png_set_compression_level, png_set_compression_strategy, png_set_filter,
    png_write_info, png_write_row, png_write_end,
    png_set_bgr,
    PNG_COLOR_TYPE_RGB, PNG_ALL_FILTERS, PNG_FILTER_NONE
)

cdef int Z_HUFFMAN_ONLY = 2  # typical zlib constant

# ---------------------------------------------------------
# Logger setup
# ---------------------------------------------------------
logger = logging.getLogger("WriteImages")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------
# Low-level streaming PNG writer (GIL held)
# ---------------------------------------------------------
cdef void _write_png_stream(
        const char* fname,
        uint8_t* base_ptr,
        int width,
        int height,
        int stride,
        int compression_level,
        int compress_strategy,
        bint disable_filters=True
):
    """Streams PNG rows to disk without allocating row pointers (zero-copy)."""

    cdef FILE* fp = fopen(fname, b"wb")
    if fp == NULL:
        return  # caller handles error

    cdef png_structp png_ptr = png_create_write_struct(b"1.6.40", NULL, NULL, NULL)
    if png_ptr == NULL:
        fclose(fp)
        return

    cdef png_infop info_ptr = png_create_info_struct(png_ptr)
    if info_ptr == NULL:
        png_destroy_write_struct(&png_ptr, NULL)
        fclose(fp)
        return

    # Initialize PNG IO and header
    png_init_io(png_ptr, fp)
    png_set_IHDR(
        png_ptr,
        info_ptr,
        <uint32_t> width,
        <uint32_t> height,
        8,                      # bit depth
        PNG_COLOR_TYPE_RGB,
        0, 0, 0
    )

    # Compression and filter settings
    png_set_compression_level(png_ptr, compression_level)
    png_set_compression_strategy(png_ptr, compress_strategy)
    if disable_filters:
        png_set_filter(png_ptr, 0, PNG_FILTER_NONE)
    else:
        png_set_filter(png_ptr, PNG_ALL_FILTERS, PNG_FILTER_NONE)

    # BGR→RGB swap zero-copy
    png_set_bgr(png_ptr)

    png_write_info(png_ptr, info_ptr)

    # Stream rows directly from memory
    cdef int y
    cdef uint8_t* rowptr
    for y in range(height):
        rowptr = <uint8_t*> (base_ptr + <size_t>(y * stride))
        png_write_row(png_ptr, rowptr)

    png_write_end(png_ptr, info_ptr)
    png_destroy_write_struct(&png_ptr, &info_ptr)
    fclose(fp)


# ---------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------
cdef int _write_png_wrapper(
        bytes fname_b,
        uint8_t* base_ptr,
        int width,
        int height,
        int stride,
        int compression_level,
        int compress_strategy,
        bint disable_filters=True
):
    """Wrapper for _write_png_stream; GIL held."""
    cdef const char* c_fname = fname_b
    try:
        _write_png_stream(c_fname, base_ptr, width, height, stride,
                          compression_level, compress_strategy,
                          disable_filters)
    except Exception:
        return -1
    return 0


# ---------------------------------------------------------
# Main worker: write_images
# ---------------------------------------------------------
def write_images(
        bytes shm_name,
        bytes desc_name,
        int frames_total,
        int width,
        int height,
        bytes output_path,
        int compression_level=1,
        int compress_strategy=Z_HUFFMAN_ONLY,
        bint disable_filters=True
):
    """Write frames from shared memory to PNG files (fast, zero-copy)."""

    cdef str shm_str = shm_name.decode()
    cdef str desc_str = desc_name.decode()
    cdef str base = output_path.decode()

    cdef int i
    cdef int stride = width * 3
    cdef int img_bytes, frame_count, bracket_index
    cdef uint8_t* data_ptr
    cdef np.ndarray contig = None
    cdef object frame_arr
    cdef bytes fname_b
    cdef uint8_t[:, :, :] mv

    logger.info(f"write_images: frames_total={frames_total} width={width} height={height}")

    shm = shared_memory.SharedMemory(name=shm_str)
    try:
        buf = np.ndarray((frames_total, height, width, 3), dtype=np.uint8, buffer=shm.buf)

        desc_shm = shared_memory.SharedMemory(name=desc_str)
        try:
            desc_arr = np.ndarray((frames_total, 3), dtype=np.uint32, buffer=desc_shm.buf)
        except Exception:
            desc_shm.close()
            raise

        for i in range(frames_total):
            img_bytes = int(desc_arr[i, 0])
            if img_bytes == 0:
                continue

            frame_count = int(desc_arr[i, 1])
            bracket_index = int(desc_arr[i, 2])

            fname_b = os.path.join(base, f"frame{frame_count:05d}_b{bracket_index}_s{img_bytes}.png").encode()

            frame_arr = buf[i]
            if frame_arr.flags.c_contiguous:
                mv = frame_arr
                data_ptr = &mv[0, 0, 0]
            else:
                contig = np.ascontiguousarray(frame_arr)
                data_ptr = <uint8_t*> contig.ctypes.data

            _write_png_wrapper(fname_b, data_ptr, width, height, stride,
                               compression_level, compress_strategy,
                               disable_filters)

            contig = None
    finally:
        desc_shm.close()
        shm.close()

    return 0
