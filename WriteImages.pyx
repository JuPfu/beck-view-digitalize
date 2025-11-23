# WriteImages_libpng_threaded.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

"""
Threaded libpng writer for shared memory frames with compression-level control.

API:
    cpdef void write_images_libpng_threaded(
        str shm_name,
        str desc_name,
        int frames_total,
        int img_width,
        int img_height,
        object output_path,
        int compression_level=6,
        int max_workers=4
    )
"""

from multiprocessing import shared_memory
from pathlib import Path
cimport numpy as cnp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging, sys

# C runtime
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fopen, fclose, fflush

# libpng API (ensure libpng dev headers available)
cdef extern from "png.h":
    ctypedef unsigned char png_byte
    ctypedef png_byte* png_bytep
    ctypedef png_bytep* png_bytepp
    ctypedef void* png_structp
    ctypedef void* png_infop

    png_structp png_create_write_struct(const char* user_png_ver, void* error_ptr, void* error_fn, void* warn_fn)
    png_infop png_create_info_struct(png_structp png_ptr)
    void png_init_io(png_structp png_ptr, FILE* fp)
    void png_set_IHDR(png_structp png_ptr, png_infop info_ptr,
                      int width, int height,
                      int bit_depth, int color_type,
                      int interlace_method, int compression_method, int filter_method)
    void png_set_rows(png_structp png_ptr, png_infop info_ptr, png_bytepp row_pointers)
    void png_write_info(png_structp png_ptr, png_infop info_ptr)
    void png_write_row(png_structp png_ptr, png_bytep row)
    void png_write_end(png_structp png_ptr, png_infop info_ptr)
    void png_destroy_write_struct(png_structp* png_ptr_ptr, png_infop* info_ptr_ptr)
    void png_set_compression_level(png_structp png_ptr, int level)

# PNG constants
cdef int PNG_COLOR_TYPE_RGB = 2
cdef int PNG_INTERLACE_NONE = 0
cdef int PNG_COMPRESSION_TYPE_BASE = 0
cdef int PNG_FILTER_TYPE_BASE = 0

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WriteImages_libpng")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


# ---------------------------------------------------------------------
# encode_png: given filename (bytes) and pointer to RGB data, write PNG via libpng
# - Runs WITH GIL (def), so libpng/fopen allowed.
# - filename: bytes (utf-8 encoded)
# - rgb_base: unsigned char* pointer to first pixel
# ---------------------------------------------------------------------
def encode_png(bytes filename,
               unsigned char* rgb_base,
               int width, int height, int stride,
               int compression_level):
    cdef png_bytep* row_ptrs = NULL
    cdef int y
    cdef FILE* fp = NULL
    cdef png_structp png_ptr = NULL
    cdef png_infop info_ptr = NULL

    # allocate array of row pointers
    row_ptrs = <png_bytep*> malloc(<size_t>height * sizeof(png_bytep))
    if row_ptrs == NULL:
        logger.error("[encode_png] malloc(row_ptrs) failed")
        return

    try:
        # populate row pointers with pointer arithmetic (C-level)
        for y in range(height):
            row_ptrs[y] = <png_bytep>(rgb_base + y * stride)

        # open file
        fp = fopen(filename, "wb")
        if fp == NULL:
            logger.error(f"[encode_png] fopen failed for {filename!r}")
            return

        # create libpng structs
        png_ptr = png_create_write_struct("1.6.40".encode('ascii'), NULL, NULL, NULL)
        if png_ptr == NULL:
            logger.error("[encode_png] png_create_write_struct failed")
            fclose(fp)
            return

        info_ptr = png_create_info_struct(png_ptr)
        if info_ptr == NULL:
            logger.error("[encode_png] png_create_info_struct failed")
            png_destroy_write_struct(&png_ptr, NULL)
            fclose(fp)
            return

        png_init_io(png_ptr, fp)

        png_set_IHDR(
            png_ptr, info_ptr,
            width, height,
            8,  # bit depth
            PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_BASE,
            PNG_FILTER_TYPE_BASE
        )

        # best-effort set compression level (libpng may provide this symbol)
        try:
            png_set_compression_level(png_ptr, compression_level)
        except Exception:
            # ignore if not available
            pass

        png_set_rows(png_ptr, info_ptr, row_ptrs)
        png_write_info(png_ptr, info_ptr)

        # write each row
        for y in range(height):
            png_write_row(png_ptr, row_ptrs[y])

        png_write_end(png_ptr, info_ptr)

    except Exception as e:
        # log and continue
        logger.error(f"[encode_png] exception while writing {filename!r}: {e}")

    finally:
        # cleanup libpng and file
        try:
            if png_ptr != NULL:
                png_destroy_write_struct(&png_ptr, &info_ptr)
        except Exception:
            pass
        try:
            if fp != NULL:
                fflush(fp)
                fclose(fp)
        except Exception:
            pass
        if row_ptrs != NULL:
            free(row_ptrs)
            row_ptrs = NULL


# ---------------------------------------------------------------------
# main threaded writer entrypoint
# ---------------------------------------------------------------------
cpdef void write_images_libpng_threaded(str shm_name,
                                        str desc_name,
                                        int frames_total,
                                        int img_width,
                                        int img_height,
                                        object output_path,
                                        int compression_level=6,
                                        int max_workers=4):
    """
    Map shared memory and submit encode tasks to a ThreadPoolExecutor.

    SHM layout:
      - pixel SHM: 1D uint8 ndarray size = frames_total * img_bytes
      - desc SHM:  uint32 ndarray shape = (frames_total, 3): (stored_bytes, frame_count, bracket_index)
    """
    cdef:
        object shm = None
        object dshm = None
        object out_path_obj
        object pixel_obj = None
        object desc_obj = None
        unsigned char* base_ptr = NULL
        int img_bytes = img_width * img_height * 3
        int i, stored_bytes, frame_count, bracket_index
        int offset, stride
        list futures = []
        bytes filename_bytes

    # normalize output path (accept Path-like or str)
    try:
        out_path_obj = Path(output_path)
    except Exception:
        out_path_obj = Path(str(output_path))

    # attach SHM segments
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        dshm = shared_memory.SharedMemory(name=desc_name, create=False)
    except Exception as e:
        logger.error(f"[write_images_libpng_threaded] SHM attach failed: {e}")
        return

    try:
        # create numpy views (no-copy) from the SHM buffers
        # pixel: 1D uint8 array of length frames_total * img_bytes
        pixel_obj = np.frombuffer(shm.buf, dtype=np.uint8, count=frames_total * img_bytes)
        # desc: uint32 flat then reshape
        desc_obj = np.frombuffer(dshm.buf, dtype=np.uint32, count=frames_total * 3).reshape((frames_total, 3))
    except Exception as e:
        logger.error(f"[write_images_libpng_threaded] mapping SHM to numpy failed: {e}")
        try:
            shm.close()
        except Exception:
            pass
        try:
            dshm.close()
        except Exception:
            pass
        return

    # ensure pixel buffer is contiguous
    if not pixel_obj.flags.c_contiguous:
        pixel_obj = np.ascontiguousarray(pixel_obj)

    # get base pointer to pixel bytes (C pointer)
    base_ptr = <unsigned char*> (<size_t> pixel_obj.ctypes.data)

    stride = img_width * 3

    # submit encoding tasks to threadpool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(frames_total):
            stored_bytes = int(desc_obj[i, 0])
            if stored_bytes == 0:
                continue
            if stored_bytes != img_bytes:
                logger.error(f"[write_images_libpng_threaded] size mismatch slot={i} stored={stored_bytes} expected={img_bytes}; skipping")
                continue

            frame_count = int(desc_obj[i, 1])
            bracket_index = int(desc_obj[i, 2])

            offset = i * img_bytes
            img_ptr = base_ptr + offset

            suffix = ('a','b','c')[bracket_index] if 0 <= bracket_index <= 2 else 'a'
            filename_bytes = str(out_path_obj / f"frame{frame_count:05d}{suffix}.png").encode('utf-8')

            # pass primitive args and C pointer to thread function
            futures.append(executor.submit(encode_png,
                                           filename_bytes,
                                           img_ptr,
                                           img_width,
                                           img_height,
                                           stride,
                                           compression_level))

        # wait for completion and log exceptions
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"[write_images_libpng_threaded] thread exception: {e}")

    # cleanup SHM handles (do not unlink; caller owns that)
    try:
        shm.close()
    except Exception:
        pass
    try:
        dshm.close()
    except Exception:
        pass

    return
