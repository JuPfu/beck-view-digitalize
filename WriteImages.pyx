# WriteImages.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
Zero-copy libpng writer for frame batches in shared memory.
This implementation:
 - uses libpng to stream PNG rows with png_write_row (no per-frame row array allocation)
 - tells libpng to swap BGR -> RGB using png_set_bgr() (zero-copy)
 - avoids copies when the numpy slice is already C-contiguous; otherwise makes one contiguous copy
 - uses multiprocessing.shared_memory as the image transport (shape: frames, H, W, 3 or 4)

Requires a matching WriteImages.pxd that declares the libpng symbols used here.
"""

import os
import sys

cimport numpy as np
import numpy as np

import gc

from libc.stdio cimport FILE, fopen, fclose
from libc.stdint cimport uint8_t, uint32_t
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free

from multiprocessing import shared_memory

# libpng API (declared in WriteImages.pxd)
from WriteImages cimport (
    png_structp, png_infop,
    png_create_write_struct, png_create_info_struct, png_destroy_write_struct,
    png_init_io, png_set_IHDR, png_set_compression_level, png_set_compression_strategy,
    png_set_filter, png_write_info, png_write_row, png_write_end,
    png_set_bgr,
    PNG_COLOR_TYPE_RGB, PNG_ALL_FILTERS, PNG_FILTER_NONE
)

cdef int Z_HUFFMAN_ONLY = 2  # typical zlib constant

# Logging helper (lightweight)
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


cdef void _write_png_libpng(const char *c_fname,
                             uint8_t *data_ptr,
                             int width,
                             int height,
                             int stride_bytes,
                             int compression_level,
                             int compress_strategy,
                             bint disable_filters=True) except *:
    """
    Write a single PNG file using libpng. This function **runs with the GIL**.
    - c_fname: UTF-8 NUL-terminated path
    - data_ptr: pointer to contiguous image data
    - stride_bytes: number of bytes per row
    """
    cdef FILE* fp = NULL
    cdef png_structp png_ptr = NULL
    cdef png_infop info_ptr = NULL
    cdef int color_type = PNG_COLOR_TYPE_RGB
    cdef int y
    cdef unsigned char *rowptr

    fp = fopen(c_fname, b"wb")
    if fp == NULL:
        raise IOError(f"fopen failed for {c_fname!s}")

    png_ptr = png_create_write_struct(b"1.6.40", NULL, NULL, NULL)
    if png_ptr == NULL:
        fclose(fp)
        raise MemoryError("png_create_write_struct failed")

    info_ptr = png_create_info_struct(png_ptr)
    if info_ptr == NULL:
        png_destroy_write_struct(&png_ptr, NULL)
        fclose(fp)
        raise MemoryError("png_create_info_struct failed")

    try:
        png_init_io(png_ptr, fp)

        png_set_IHDR(png_ptr, info_ptr,
                     <uint32_t> width,
                     <uint32_t> height,
                     8,  # bit depth
                     color_type,
                     0, 0, 0)

        # compression tuning
        png_set_compression_level(png_ptr, compression_level)
        png_set_compression_strategy(png_ptr, compress_strategy)

        # filtering (fastest when disabled)
        if disable_filters:
            png_set_filter(png_ptr, 0, PNG_FILTER_NONE)
        else:
            png_set_filter(png_ptr, PNG_ALL_FILTERS, PNG_FILTER_NONE)

        # BGR -> RGB swap (zero-copy)
        png_set_bgr(png_ptr)

        png_write_info(png_ptr, info_ptr)

        # stream rows directly from data_ptr (no row pointer array)
        for y in range(height):
            rowptr = <unsigned char*> (data_ptr + <size_t>(y * stride_bytes))
            png_write_row(png_ptr, rowptr)

        png_write_end(png_ptr, info_ptr)

    finally:
        if png_ptr != NULL:
            png_destroy_write_struct(&png_ptr, &info_ptr)
        if fp != NULL:
            fclose(fp)


# ----------------------
# Public worker
# ----------------------

def write_images(bytes shm_name,
                 bytes desc_name,
                 int frames_total,
                 int width,
                 int height,
                 bytes output_path,
                 int compression_level=1,
                 int compress_strategy=Z_HUFFMAN_ONLY,
                 bint disable_filters=True):
    """
    Worker entrypoint that maps the shared-memory buffer and writes PNG files.

    - shm_name, desc_name, output_path: bytes (shared memory names / directory)
    - frames_total, width, height: ints
    - compression_level: libpng 0..9 (lower = faster)
    - compress_strategy: zlib strategy (e.g. Z_HUFFMAN_ONLY); default is 1 (sane default)
    - disable_filters: if True sets PNG_FILTER_NONE (fastest)

    Behavior:
    - For each frame: if the numpy view is C-contiguous we take its address (zero-copy).
      Otherwise we make a single contiguous copy (np.ascontiguousarray) and use that.
    - Filenames are written as bytes to avoid encoding problems between processes.
    """

    cdef str shm_str = shm_name.decode()
    cdef str desc_str = desc_name.decode()
    cdef str base = output_path.decode()

    cdef int i
    cdef int img_bytes, frame_count, bracket_index
    cdef int stride = width * 3
    cdef uint8_t *data_ptr = NULL

    cdef np.ndarray buf = None
    cdef np.ndarray desc = None
    cdef object frame_slice
    cdef np.ndarray contig = None

    tuple suffix_map = ('a', 'b', 'c')

    cdef np.uint8_t[:, :, :] mv

    logger.debug(f"write_images: frames_total={frames_total} width={width} height={height} base={base}")

    # Attach shared memory
    shm = shared_memory.SharedMemory(name=shm_str)
    desc_shm = shared_memory.SharedMemory(name=desc_str)

    try:
        buf = np.ndarray((frames_total, height, width, 3), dtype=np.uint8, buffer=shm.buf)

        try:
            desc = np.ndarray((frames_total, 3), dtype=np.uint32, buffer=desc_shm.buf)
        except Exception:
            desc_shm.close()
            raise

        for i in range(frames_total):
            img_bytes = int(desc[i, 0])
            if img_bytes == 0:
                continue

            frame_count = int(desc[i, 1])
            bracket_index = int(desc[i, 2])

            # suffix from bracket_index (fallback to 'a'.. if out of range)
            if 0 <= bracket_index < len(suffix_map):
                suffix = suffix_map[bracket_index]
            else:
                suffix = 'a'

            fname = os.path.join(base, f"frame{frame_count:05d}{suffix}.png")
            c_fname = fname.encode('utf-8')

            # get slice
            frame_slice = buf[i]

            # fast path: zero-copy if already C-contiguous
            if frame_slice.flags.c_contiguous and frame_slice.dtype == np.uint8:
                # create a typed memoryview referencing the ndarray so we can take its address
                mv = frame_slice
                data_ptr = &mv[0, 0, 0]
                # call writer (GIL-held)
                _write_png_libpng(<const char*> c_fname, data_ptr, width, height, stride,
                                  compression_level, compress_strategy, disable_filters)

            else:
                # single contiguous copy then write
                contig = np.ascontiguousarray(frame_slice)
                data_ptr = <uint8_t *> contig.ctypes.data
                _write_png_libpng(<const char*> c_fname, data_ptr, width, height, stride,
                                  compression_level, compress_strategy, disable_filters)
                # release reference so memory can be freed
                contig = None
    finally:
        # drop numpy views explicitly
        try:
            del buf
        except Exception:
            pass
        try:
            del desc
        except Exception:
            pass

        # force GC so exported pointers are released
        try:
            gc.collect()
        except Exception:
            pass

        # close shared memory (do NOT unlink if parent owns it)
        try:
            shm.close()
        except Exception:
            pass
        try:
            desc_shm.close()
        except Exception:
            pass

    return 0
