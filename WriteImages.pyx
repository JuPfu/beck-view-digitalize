# WriteImages.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import logging

import os
import sys
cimport numpy as np
import numpy as np

from libc.stdio cimport FILE, fopen, fclose
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free

from multiprocessing import shared_memory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

# libpng API
from WriteImages cimport (
    png_structp, png_infop,
    png_create_write_struct, png_create_info_struct,
    png_destroy_write_struct,
    png_init_io, png_set_IHDR,
    png_set_compression_level,
    png_write_info, png_write_image, png_write_end,
    PNG_COLOR_TYPE_RGB
)

# ---------------------------------------------------------
# Optional BGR→RGB swap
# ---------------------------------------------------------
cdef inline void swap_bgr(uint8_t* row, int width) noexcept:
    cdef int x
    cdef uint8_t tmp
    for x in range(width):
        tmp = row[3*x]
        row[3*x] = row[3*x + 2]
        row[3*x + 2] = tmp


# ---------------------------------------------------------
# Low-level PNG write using libpng
# ---------------------------------------------------------
cdef void write_png(
    const char* fname,
    uint8_t* data_ptr,
    int width,
    int height,
    int stride,
    int compression_level
):
    cdef FILE* fp
    cdef png_structp png_ptr
    cdef png_infop info_ptr
    cdef unsigned char** rows = NULL
    cdef int y

    fp = fopen(fname, b"wb")
    if fp == NULL:
        raise IOError(f"Cannot open {fname!s}")

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

        png_set_IHDR(
            png_ptr, info_ptr,
            width, height,
            8,                          # bit depth
            PNG_COLOR_TYPE_RGB,         # always RGB
            0, 0, 0
        )

        png_set_compression_level(png_ptr, compression_level)

        png_write_info(png_ptr, info_ptr)

        rows = <unsigned char**> malloc(height * sizeof(unsigned char*))
        if rows == NULL:
            raise MemoryError("row_pointers malloc failed")

        # prepare row pointers
        for y in range(height):
            rows[y] = data_ptr + y * stride
            swap_bgr(rows[y], width)

        png_write_image(png_ptr, rows)
        png_write_end(png_ptr, info_ptr)

    finally:
        if rows != NULL:
            free(rows)
        png_destroy_write_struct(&png_ptr, &info_ptr)
        fclose(fp)


# ---------------------------------------------------------
# Main worker using multiprocessing.SharedMemory
# ---------------------------------------------------------
def write_images(
    bytes shm_name,
    bytes desc_name,
    int frames_total,
    int width,
    int height,
    bytes output_path,
    int compression_level
):

    cdef str shm_str = shm_name.decode()
    cdef str desc_str = desc_name.decode()

    cdef str base = output_path.decode()

    cdef bytes filename

    # Used to build filenames
    cdef int img_bytes, frame_count, bracket_index

    cdef uint8_t[:,:,:,:] mv
    cdef uint8_t* frame_ptr

    # ----------------------------------------------------------
    # Attach image shared memory buffer
    # Shared memory shape is (frames, height, width, 3)
    # ----------------------------------------------------------
    shm = shared_memory.SharedMemory(name=shm_str)
    buf = np.ndarray((frames_total, height, width, 3), dtype=np.uint8, buffer=shm.buf)

    mv = buf

    # ----------------------------------------------------------
    # Attach descriptor shared memory buffer
    # Layout guaranteed by DigitizeVideo.pyx:
    #   desc_arr[i,0] = img_bytes (uint32)
    #   desc_arr[i,1] = frame_count (uint32)
    #   desc_arr[i,2] = bracket_index (uint32)
    # ----------------------------------------------------------
    desc_shm = shared_memory.SharedMemory(name=desc_str)
    desc_arr = np.ndarray((frames_total, 3), dtype=np.uint32, buffer=desc_shm.buf)

    cdef int stride = width * 3
    cdef int i

    for i in range(frames_total):

        # Extract descriptor metadata
        img_bytes     = int(desc_arr[i, 0])
        frame_count   = int(desc_arr[i, 1])
        bracket_index = int(desc_arr[i, 2])

        # Build descriptive filename
        filename = os.path.join(base, f"frame{frame_count:05d}_b{bracket_index}_s{img_bytes}.png").encode()

        logger.debug(f"Write PNG {filename=}")

        frame_ptr = &mv[i, 0, 0, 0]

        write_png(filename, frame_ptr, width, height, stride, compression_level)

    shm.close()
    desc_shm.close()

    return 0
