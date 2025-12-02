# WriteImages.pyx
# cython: language_level=3, boundscheck=False, wraparound=False

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import os
cimport numpy as np
import numpy as np

from libc.stdio cimport FILE, fopen, fclose
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

from WriteImages cimport (
    png_create_write_struct, png_create_info_struct, png_destroy_write_struct,
    png_init_io, png_set_IHDR, png_set_compression_level,
    png_write_info, png_write_image, png_write_end,
    PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGBA
)

from multiprocessing import shared_memory

cdef inline void swap_bgr_channels(uint8_t[:, :, :] mv) noexcept:
    cdef int y, x
    cdef uint8_t t
    for y in range(mv.shape[0]):
        for x in range(mv.shape[1]):
            t = mv[y, x, 0]
            mv[y, x, 0] = mv[y, x, 2]
            mv[y, x, 2] = t

def _write_png_from_rgb_file(const char* fname, uint8_t* data, int width, int height, int stride, bint has_alpha):
    """
    Write PNG using libpng from a raw contiguous buffer.
    - data: pointer to contiguous pixel data (row-major).
    - stride: bytes per row
    - has_alpha: if True expects RGBA with stride == width*4, else RGB width*3
    """
    cdef FILE* fp = NULL
    cdef png_structp png_ptr = NULL
    cdef png_infop info_ptr = NULL
    cdef int color_type = PNG_COLOR_TYPE_RGB
    cdef int bit_depth = 8
    cdef int y
    cdef unsigned char **row_pointers = NULL

    if has_alpha:
        color_type = PNG_COLOR_TYPE_RGBA

    fp = fopen(fname, b"wb")
    if fp == NULL:
        raise IOError(f"fopen failed for {fname!r}")

    png_ptr = png_create_write_struct(b"1.6.37", NULL, NULL, NULL)
    if png_ptr == NULL:
        fclose(fp)
        raise MemoryError("png_create_write_struct failed")

    info_ptr = png_create_info_struct(png_ptr)
    if info_ptr == NULL:
        png_destroy_write_struct(&png_ptr, NULL)
        fclose(fp)
        raise MemoryError("png_create_info_struct failed")

    # Initialize IO and set basic params
    png_init_io(png_ptr, fp)
    png_set_IHDR(png_ptr, info_ptr, <uint32_t>width, <uint32_t>height,
                 bit_depth, color_type,
                 0, 0, 0)
    # default compression level 6; adjust if needed
    png_set_compression_level(png_ptr, 6)

    # prepare row pointers (point into data)
    # allocate array of pointers
    row_pointers = <unsigned char **> PyMem_Malloc(height * sizeof(unsigned char *))
    if row_pointers == NULL:
        png_destroy_write_struct(&png_ptr, &info_ptr)
        fclose(fp)
        raise MemoryError("out of memory for row pointers")

    try:
        for y in range(height):
            # each row is data + y * stride
            row_pointers[y] = <unsigned char *> (data + y * stride)

        png_write_info(png_ptr, info_ptr)
        png_write_image(png_ptr, row_pointers)
        png_write_end(png_ptr, info_ptr)
    finally:
        if row_pointers != NULL:
            PyMem_Free(row_pointers)
        if png_ptr != NULL:
            png_destroy_write_struct(&png_ptr, &info_ptr)
        if fp != NULL:
            fclose(fp)


def write_images(bytes shm_name,
                 bytes desc_name,
                 int frames_total,
                 int width,
                 int height,
                 bytes output_path,
                 int compression_level,
                 bint incoming_is_bgr=False,
                 bint input_has_alpha=False):
    """
    Worker entrypoint writes frames to PNG files using libpng.

    incoming_is_bgr: if True perform BGR->RGB swap.
    input_has_alpha: if True input buffer is RGBA; output will be RGBA.
    """

    cdef int i, img_bytes, frame_count, bracket_index
    cdef uint8_t* frame_ptr
    cdef np.uint8_t[:, :, :] frame_mv
    cdef np.ndarray buf = None
    cdef np.ndarray desc = None
    cdef int stride_bytes

    if not isinstance(shm_name, (bytes, bytearray)):
        raise TypeError("shm_name must be bytes")
    if not isinstance(desc_name, (bytes, bytearray)):
        raise TypeError("desc_name must be bytes")
    if not isinstance(output_path, (bytes, bytearray)):
        raise TypeError("output_path must be bytes")

    base = output_path.decode()

    try:
        shm = shared_memory.SharedMemory(name=shm_name.decode())
        buf = np.ndarray((frames_total, height, width, 4 if input_has_alpha else 3),
                         dtype=np.uint8, buffer=shm.buf)

        desc_shm = shared_memory.SharedMemory(name=desc_name.decode())
        desc = np.ndarray((frames_total, 3), dtype=np.uint32, buffer=desc_shm.buf)

        try:
            import logging
            logging.getLogger("WriteImages").info(f"write_images: frames_total={frames_total}, width={width}, height={height}")
        except Exception:
            pass

        for i in range(frames_total):
            img_bytes = int(desc[i, 0])
            if img_bytes == 0:
                continue

            frame_count = int(desc[i, 1])
            bracket_index = int(desc[i, 2])
            filename = os.path.join(base, f"frame{frame_count:05d}_b{bracket_index}.png")

            if img_bytes != (width * height * (4 if input_has_alpha else 3)):
                import logging
                logging.getLogger("WriteImages").warning(
                    f"Descriptor img_bytes ({img_bytes}) != expected ({width*height*(4 if input_has_alpha else 3)})"
                )

            # Get the frame memoryview
            frame_mv = buf[i]   # typed memoryview possible but here it's a numpy view

            # Ensure contiguous copy and correct dtype
            arr = np.ascontiguousarray(frame_mv)
            if incoming_is_bgr:
                # fast in-place swap on the contiguous copy using typed memoryview
                cmv = arr
                swap_bgr_channels(cmv)

            # Data pointer + stride
            stride_bytes = arr.strides[0]    # bytes per row
            frame_ptr = <uint8_t*> arr.ctypes.data  # get C pointer to first element

            # Write using libpng
            _write_png_from_rgb_file(filename.encode('utf-8'), frame_ptr, width, height, stride_bytes, input_has_alpha)

    except Exception:
        try:
            import traceback, logging
            logging.getLogger("WriteImages").exception("Worker write_images failed")
            traceback.print_exc()
        except Exception:
            pass
        raise
    finally:
        try:
            shm.close()
        except Exception:
            pass
        try:
            desc_shm.close()
        except Exception:
            pass
