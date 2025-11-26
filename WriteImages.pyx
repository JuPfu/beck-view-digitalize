# WriteImages.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

from multiprocessing import shared_memory
from pathlib import Path
import logging, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
cimport numpy as cnp

# C imports
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, fopen, fclose, fflush
from libc.stdint cimport uint8_t, uint16_t, uint32_t
from libc.stddef cimport size_t

# Import our structs and API from pxd
from WriteImages cimport (
    spng_ctx,
    spng_ihdr,
    spng_ctx_new,
    spng_ctx_free,
    spng_set_png_file,
    spng_set_option,
    spng_set_ihdr,
    spng_encode_image,
    spng_strerror,
    spng_version_string
)

# constants (from spng.h)
cdef int SPNG_OK = 0
cdef int SPNG_FMT_RGB8 = 4
cdef int SPNG_ENCODE_FINALIZE = 2
cdef int SPNG_IMG_COMPRESSION_LEVEL = 2

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WriteImages_spng")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


# ---------------------------------------------------------------------
# encode_png_spng — called inside threads
# ---------------------------------------------------------------------
def encode_png_spng(bytes filename,
                    unsigned char* img_ptr,
                    int width, int height, int stride,
                    int compression_level):
    cdef spng_ctx* ctx = NULL
    cdef spng_ihdr ihdr
    cdef int rc
    cdef size_t img_len
    cdef FILE* fp = NULL

    img_len = <size_t> height * <size_t> stride

    fp = fopen(filename, "wb")
    if fp == NULL:
        logger.error(f"[encode_png_spng] fopen failed for {filename!r}")
        return

    ctx = spng_ctx_new(0)
    if ctx == NULL:
        logger.error("[encode_png_spng] spng_ctx_new failed")
        fclose(fp)
        return

    rc = spng_set_png_file(ctx, fp)
    if rc != SPNG_OK:
        logger.error(f"[encode_png_spng] spng_set_png_file failed: {spng_strerror(rc).decode('utf-8')}")
        spng_ctx_free(ctx)
        fclose(fp)
        return

    # Fill IHDR
    ihdr.width  = <uint32_t> width
    ihdr.height = <uint32_t> height
    ihdr.bit_depth = <uint8_t> 8
    ihdr.color_type = <uint8_t> 2
    ihdr.compression_method = <uint8_t> 0
    ihdr.filter_method = <uint8_t> 0
    ihdr.interlace_method = <uint8_t> 0

    rc = spng_set_ihdr(ctx, &ihdr)
    if rc != SPNG_OK:
        logger.error(f"[encode_png_spng] spng_set_ihdr failed: {spng_strerror(rc).decode('utf-8')}")
        spng_ctx_free(ctx)
        fclose(fp)
        return

    rc = spng_set_option(ctx, SPNG_IMG_COMPRESSION_LEVEL, compression_level)
    # non-fatal: log only in debug mode

    rc = spng_encode_image(ctx, <const void*> img_ptr, img_len,
                           SPNG_FMT_RGB8, SPNG_ENCODE_FINALIZE)
    if rc != SPNG_OK:
        logger.error(f"[encode_png_spng] encode failed: {spng_strerror(rc).decode('utf-8')}")

    spng_ctx_free(ctx)
    try:
        fflush(fp)
    except Exception:
        pass
    fclose(fp)


# ---------------------------------------------------------------------
# write_images_spng_threaded — main entrypoint
# ---------------------------------------------------------------------
cpdef void write_images_spng_threaded(str shm_name,
                                      str desc_name,
                                      int frames_total,
                                      int img_width,
                                      int img_height,
                                      object output_path,
                                      int compression_level=6,
                                      int max_workers=4):
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
        unsigned char* img_ptr

    try:
        out_path_obj = Path(output_path)
    except Exception:
        out_path_obj = Path(str(output_path))

    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        dshm = shared_memory.SharedMemory(name=desc_name, create=False)
    except Exception as e:
        logger.error(f"[write_images_spng_threaded] failed to attach SHM: {e}")
        return

    try:
        pixel_obj = np.frombuffer(shm.buf, dtype=np.uint8, count=frames_total * img_bytes)
        desc_obj = np.frombuffer(dshm.buf, dtype=np.uint32, count=frames_total * 3).reshape((frames_total, 3))
    except Exception as e:
        logger.error(f"[write_images_spng_threaded] numpy mapping failed: {e}")
        try: shm.close()
        except: pass
        try: dshm.close()
        except: pass
        return

    if not pixel_obj.flags.c_contiguous:
        pixel_obj = np.ascontiguousarray(pixel_obj)

    base_ptr = <unsigned char*> pixel_obj.ctypes.data
    stride = img_width * 3

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(frames_total):
            stored_bytes = int(desc_obj[i, 0])
            if stored_bytes != img_bytes:
                continue

            frame_count = int(desc_obj[i, 1])
            bracket_index = int(desc_obj[i, 2])

            offset = i * img_bytes
            img_ptr = base_ptr + offset

            suffix = ('a','b','c')[bracket_index] if 0 <= bracket_index <= 2 else 'a'
            filename_bytes = str(out_path_obj / f"frame{frame_count:05d}{suffix}.png").encode("utf-8")

            futures.append(
                executor.submit(
                    encode_png_spng,
                    filename_bytes,
                    img_ptr,
                    img_width,
                    img_height,
                    stride,
                    compression_level
                )
            )

        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logger.error(f"[write_images_spng_threaded] thread error: {e}")

    try: shm.close()
    except: pass
    try: dshm.close()
    except: pass
