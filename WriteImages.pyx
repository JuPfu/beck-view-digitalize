# WriteImages.pyx
# cython: language_level=3, boundscheck=False, wraparound=False

from cpython.mem cimport PyMem_Malloc, PyMem_Free

import os
cimport numpy as np
import numpy as np

from libc.stdio cimport FILE, fopen, fclose
from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

from WriteImages cimport (
    png_create_write_struct, png_create_info_struct,
    png_destroy_write_struct, png_init_io, png_set_IHDR,
    png_set_compression_level, png_write_info,
    png_write_image, png_write_end,
    PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGBA
)

from multiprocessing import shared_memory


# ---------------------------------------------------------------------------
#  BGR <-> RGB swap for contiguous uint8 image
# ---------------------------------------------------------------------------

cdef inline void swap_bgr_channels(uint8_t[:, :, :] mv) noexcept:
    """
    In-place swap of B and R channels for a uint8 HxWx3 buffer.
    """
    cdef int y, x
    cdef uint8_t t
    for y in range(mv.shape[0]):
        for x in range(mv.shape[1]):
            t = mv[y, x, 0]
            mv[y, x, 0] = mv[y, x, 2]
            mv[y, x, 2] = t


# ---------------------------------------------------------------------------
#  Low-level wrapper: write a PNG file from a raw contiguous RGB / RGBA buffer
# ---------------------------------------------------------------------------

def _write_png_from_rgb_file(const char* fname,
                             uint8_t* data,
                             int width,
                             int height,
                             int stride,
                             bint has_alpha):
    """
    Write PNG using libpng from a raw contiguous buffer.

    Parameters
    ----------
    fname : bytes
        File name (UTF-8 encoded).
    data : uint8_t*
        Pointer to the first byte of the image buffer.
    width, height : int
    stride : int
        Bytes per row in memory.
    has_alpha : bool
        Whether buffer is RGB or RGBA.
    """

    cdef FILE* fp = NULL
    cdef png_structp png_ptr = NULL
    cdef png_infop info_ptr = NULL
    cdef unsigned char** row_pointers = NULL
    cdef int color_type = PNG_COLOR_TYPE_RGBA if has_alpha else PNG_COLOR_TYPE_RGB
    cdef int y

    # -- open file -----------------------------------------------------------
    fp = fopen(fname, b"wb")
    if fp == NULL:
        raise IOError(f"fopen failed for {fname!r}")

    # -- create PNG writer structures ---------------------------------------
    png_ptr = png_create_write_struct(b"1.6.37", NULL, NULL, NULL)
    if png_ptr == NULL:
        fclose(fp)
        raise MemoryError("png_create_write_struct failed")

    info_ptr = png_create_info_struct(png_ptr)
    if info_ptr == NULL:
        png_destroy_write_struct(&png_ptr, NULL)
        fclose(fp)
        raise MemoryError("png_create_info_struct failed")

    # -- set header info -----------------------------------------------------
    png_init_io(png_ptr, fp)
    png_set_IHDR(
        png_ptr, info_ptr,
        <uint32_t>width, <uint32_t>height,
        8,                      # bit depth
        color_type,
        0, 0, 0
    )

    # default compression level (6)
    png_set_compression_level(png_ptr, 6)

    # -- allocate row pointer array -----------------------------------------
    row_pointers = <unsigned char**> PyMem_Malloc(height * sizeof(unsigned char*))
    if row_pointers is NULL:
        png_destroy_write_struct(&png_ptr, &info_ptr)
        fclose(fp)
        raise MemoryError("Out of memory for row pointers")

    try:
        # Point each row into the contiguous data buffer
        for y in range(height):
            row_pointers[y] = <unsigned char*> (data + y * stride)

        png_write_info(png_ptr, info_ptr)
        png_write_image(png_ptr, row_pointers)
        png_write_end(png_ptr, info_ptr)

    finally:
        if row_pointers is not NULL:
            PyMem_Free(row_pointers)
        if png_ptr is not NULL:
            png_destroy_write_struct(&png_ptr, &info_ptr)
        if fp is not NULL:
            fclose(fp)



# ---------------------------------------------------------------------------
#  Worker entry point: take frames from shared memory and write PNGs
# ---------------------------------------------------------------------------

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
    Read frames from two shared memory blocks and write each frame to a PNG file.

    Shared memory layout
    --------------------
    shm_name  : contains the image data buffer
        shape: (frames_total, height, width, channels)
        channels = 3 (RGB) or 4 (RGBA)

    desc_name : contains per-frame metadata
        shape: (frames_total, 3)
        desc[i, 0] = img_bytes
        desc[i, 1] = frame_count (for filename)
        desc[i, 2] = exposure-bracket index

    Notes
    -----
    - Incoming frames may be BGR (OpenCV). If so, a fast
      in-place typed-memoryview BGR→RGB swap is applied.
    - The shared memory is closed when this function returns.
    """

    cdef int i
    cdef int img_bytes, frame_count, bracket_index
    cdef int stride_bytes
    cdef uint8_t* frame_ptr

    cdef np.ndarray buf = None
    cdef np.ndarray desc = None
    cdef np.uint8_t[:, :, :] frame_mv

    if not isinstance(shm_name, (bytes, bytearray)):
        raise TypeError("shm_name must be bytes")
    if not isinstance(desc_name, (bytes, bytearray)):
        raise TypeError("desc_name must be bytes")
    if not isinstance(output_path, (bytes, bytearray)):
        raise TypeError("output_path must be bytes")

    base = output_path.decode()

    try:
        # -- attach to shared memories --------------------------------------
        shm = shared_memory.SharedMemory(name=shm_name.decode())
        buf = np.ndarray(
            (frames_total, height, width, 4 if input_has_alpha else 3),
            dtype=np.uint8, buffer=shm.buf
        )

        desc_shm = shared_memory.SharedMemory(name=desc_name.decode())
        desc = np.ndarray((frames_total, 3), dtype=np.uint32, buffer=desc_shm.buf)

        # optional debug log
        try:
            import logging
            logging.getLogger("WriteImages").info(
                f"write_images: frames_total={frames_total}, size={width}x{height}"
            )
        except Exception:
            pass

        # -- main loop: write each frame -----------------------------------
        for i in range(frames_total):
            img_bytes = int(desc[i, 0])
            if img_bytes == 0:
                continue  # empty slot

            frame_count   = int(desc[i, 1])
            bracket_index = int(desc[i, 2])

            filename = os.path.join(
                base,
                f"frame{frame_count:05d}_b{bracket_index}.png"
            )

            # Sanity check
            expected_bytes = width * height * (4 if input_has_alpha else 3)
            if img_bytes != expected_bytes:
                import logging
                logging.getLogger("WriteImages").warning(
                    f"Descriptor img_bytes {img_bytes} != expected {expected_bytes}"
                )

            # frame_mv is a NumPy view into shared memory
            frame_mv = buf[i]

            # Ensure contiguous copy (required for libpng row-pointer layout)
            arr = np.ascontiguousarray(frame_mv)

            # Optional BGR→RGB fix (OpenCV source)
            if incoming_is_bgr:
                cdef uint8_t[:, :, :] cmv = arr
                swap_bgr_channels(cmv)

            # Data pointer + stride
            stride_bytes = arr.strides[0]
            frame_ptr = <uint8_t*> arr.ctypes.data

            # Write PNG file
            _write_png_from_rgb_file(
                filename.encode("utf-8"),
                frame_ptr, width, height, stride_bytes,
                input_has_alpha
            )

    except Exception:
        # Log and reraise
        try:
            import traceback, logging
            logging.getLogger("WriteImages").exception("Worker write_images failed")
            traceback.print_exc()
        except Exception:
            pass
        raise

    finally:
        # Always close SHM handles even on error
        try: shm.close()
        except Exception: pass
        try: desc_shm.close()
        except Exception: pass
