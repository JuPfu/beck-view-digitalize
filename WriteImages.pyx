# WriteImages.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
Zero-copy libpng writer for frame batches in shared memory.

 - streams PNG rows with png_write_row straight from the shared memory buffer (no per-frame
   row array, no copy); BGR -> RGB is done by libpng (png_set_bgr)
 - the complete libpng sequence runs in a small C helper (see the verbatim C block below).
   The helper installs its own libpng error handler with setjmp/longjmp. The previous version
   passed no handler, so any libpng error (disk full, out of memory, ...) ended in abort():
   the worker process died, the task never finished and the buffer was never released.
   Now an error becomes a Python exception, the half written file is removed and the other
   images of the chunk are still written.
 - fclose()/fflush() results are checked (a full disk shows up there for buffered output)
 - on Windows the UTF-8 file name is converted to UTF-16 (_wfopen): fopen() would interpret the
   bytes in the ANSI code page and fail for paths such as C:\\Users\\Jürgen\\Filme
 - the GIL is released while a file is written
 - shared memory is attached without resource-tracker registration where Python supports it
   (3.13+); the parent owns and unlinks the segments

The libpng declarations in WriteImages.pxd are no longer needed by this file.
"""

import os
import sys
import logging

cimport numpy as np
import numpy as np

from libc.stddef cimport size_t

from multiprocessing import shared_memory

cdef extern from *:
    """
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <errno.h>
    #include <setjmp.h>
    #include <png.h>
    #ifdef _WIN32
    #include <windows.h>
    #endif

    typedef struct { char msg[192]; } wi_error_t;

    static void wi_error_fn(png_structp png_ptr, png_const_charp msg)
    {
        wi_error_t *err = (wi_error_t *) png_get_error_ptr(png_ptr);
        if (err != NULL) {
            strncpy(err->msg, (msg != NULL) ? msg : "libpng error", sizeof(err->msg) - 1);
            err->msg[sizeof(err->msg) - 1] = '\\0';
        }
        longjmp(png_jmpbuf(png_ptr), 1);
    }

    static void wi_warning_fn(png_structp png_ptr, png_const_charp msg)
    {
        (void) png_ptr;
        (void) msg;
    }

    #ifdef _WIN32
    static wchar_t *wi_widen(const char *utf8)
    {
        wchar_t *w;
        int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
        if (n <= 0) return NULL;
        w = (wchar_t *) malloc((size_t) n * sizeof(wchar_t));
        if (w == NULL) return NULL;
        if (MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w, n) <= 0) { free(w); return NULL; }
        return w;
    }
    #endif

    static FILE *wi_fopen_write(const char *utf8_name)
    {
    #ifdef _WIN32
        FILE *fp = NULL;
        wchar_t *w = wi_widen(utf8_name);
        if (w == NULL) return NULL;
        fp = _wfopen(w, L"wb");
        free(w);
        return fp;
    #else
        return fopen(utf8_name, "wb");
    #endif
    }

    static void wi_remove_file(const char *utf8_name)
    {
    #ifdef _WIN32
        wchar_t *w = wi_widen(utf8_name);
        if (w != NULL) { _wremove(w); free(w); }
    #else
        remove(utf8_name);
    #endif
    }

    /* Write one 8 bit RGB(BGR in memory) PNG. Returns 0 on success, otherwise an error code
       and a message in errbuf. Never calls abort() because of libpng errors. */
    static int wi_write_png(const char *fname, const unsigned char *data,
                            int width, int height, int stride,
                            int level, int strategy, int disable_filters,
                            char *errbuf, size_t errlen)
    {
        FILE *volatile fp = NULL;
        png_structp png_ptr = NULL;
        png_infop info_ptr = NULL;
        wi_error_t err;
        int rc = 0;
        int y;

        err.msg[0] = '\\0';
        if (errbuf != NULL && errlen > 0) errbuf[0] = '\\0';

        fp = wi_fopen_write(fname);
        if (fp == NULL) {
            snprintf(errbuf, errlen, "cannot open file for writing: %s", strerror(errno));
            return 1;
        }
        /* fully buffered I/O with a large buffer */
        setvbuf(fp, NULL, _IOFBF, 1 << 20);

        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, &err, wi_error_fn, wi_warning_fn);
        if (png_ptr == NULL) {
            snprintf(errbuf, errlen, "png_create_write_struct failed");
            rc = 2;
            goto done;
        }
        info_ptr = png_create_info_struct(png_ptr);
        if (info_ptr == NULL) {
            snprintf(errbuf, errlen, "png_create_info_struct failed");
            rc = 3;
            goto done;
        }

        if (setjmp(png_jmpbuf(png_ptr))) {
            /* libpng reported an error (e.g. write error because the disk is full) */
            snprintf(errbuf, errlen, "%s", err.msg[0] ? err.msg : "libpng error");
            rc = 4;
            goto done;
        }

        png_init_io(png_ptr, fp);
        png_set_IHDR(png_ptr, info_ptr, (png_uint_32) width, (png_uint_32) height, 8,
                     PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
        png_set_compression_level(png_ptr, level);
        png_set_compression_strategy(png_ptr, strategy);
        png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE,
                       disable_filters ? PNG_FILTER_NONE : PNG_ALL_FILTERS);
        png_set_bgr(png_ptr);  /* BGR (OpenCV) -> RGB without copying */
        png_write_info(png_ptr, info_ptr);

        for (y = 0; y < height; y++) {
            png_write_row(png_ptr, (png_bytep) (data + (size_t) y * (size_t) stride));
        }
        png_write_end(png_ptr, info_ptr);

    done:
        png_destroy_write_struct(&png_ptr, &info_ptr);

        if (rc == 0 && (fflush(fp) != 0 || ferror(fp))) {
            snprintf(errbuf, errlen, "write failed: %s", strerror(errno));
            rc = 5;
        }
        if (fclose(fp) != 0 && rc == 0) {
            snprintf(errbuf, errlen, "closing the file failed: %s", strerror(errno));
            rc = 6;
        }
        fp = NULL;

        if (rc != 0) {
            wi_remove_file(fname);  /* do not leave truncated PNG files behind */
        }
        return rc;
    }
    """
    int wi_write_png(const char *fname, const unsigned char *data,
                     int width, int height, int stride,
                     int level, int strategy, int disable_filters,
                     char *errbuf, size_t errlen) nogil


# Logging helper (lightweight)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _attach(name):
    """Attach to a shared memory segment owned by the parent process."""
    try:
        # Python >= 3.13: do not register the segment with the resource tracker
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


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
                 int compress_strategy=1,
                 bint disable_filters=True):
    """
    Worker entrypoint that maps the shared-memory buffer and writes PNG files.

    - shm_name, desc_name, output_path: bytes (shared memory names / directory)
    - frames_total, width, height: ints
    - compression_level: libpng 0..9 (lower = faster)
    - compress_strategy: zlib strategy (default 1 = Z_FILTERED)
    - disable_filters: if True sets PNG_FILTER_NONE (fastest)

    The descriptor array has one row per image: (bytes, frame count, bracket index); a row with
    0 bytes is empty. All descriptors are cleared at the end (also after errors) so that the
    buffer can be reused.

    Errors writing single files are logged and do not stop the other images; after the loop a
    RuntimeError reports how many images failed (the parent logs it).
    """

    cdef str shm_str = shm_name.decode()
    cdef str desc_str = desc_name.decode()
    cdef str base = output_path.decode()

    cdef int i
    cdef int img_bytes, frame_count, bracket_index
    cdef int stride = width * 3
    cdef int expected_bytes = width * height * 3
    cdef int rc = 0
    cdef int failures = 0

    cdef const unsigned char *data_ptr = NULL
    cdef const char *c_name = NULL
    cdef bytes c_fname
    cdef bytes msg_b
    cdef char errbuf[256]

    cdef np.ndarray buf = None
    cdef np.ndarray desc = None
    cdef np.uint32_t[:, :] desc_mv
    cdef np.uint8_t[:, :, :] mv

    cdef tuple suffix_map = ('a', 'b', 'c')

    logger.debug(f"write_images: frames_total={frames_total} width={width} height={height} base={base}")

    shm = None
    desc_shm = None
    first_error = None

    try:
        # Attach shared memory
        shm = _attach(shm_str)
        desc_shm = _attach(desc_str)

        buf = np.ndarray((frames_total, height, width, 3), dtype=np.uint8, buffer=shm.buf)
        desc = np.ndarray((frames_total, 3), dtype=np.uint32, buffer=desc_shm.buf)
        desc_mv = desc

        for i in range(frames_total):
            img_bytes = desc_mv[i, 0]
            if img_bytes == 0:
                continue

            frame_count = desc_mv[i, 1]
            bracket_index = desc_mv[i, 2]

            if img_bytes != expected_bytes:
                failures += 1
                if first_error is None:
                    first_error = f"image {frame_count}: descriptor says {img_bytes} bytes, expected {expected_bytes}"
                logger.error(f"write_images: {first_error}")
                continue

            # suffix from bracket_index (fallback to 'a' if out of range)
            if 0 <= bracket_index < len(suffix_map):
                suffix = suffix_map[bracket_index]
            else:
                suffix = 'a'

            fname = os.path.join(base, f"frame{frame_count:05d}{suffix}.png")
            c_fname = fname.encode('utf-8')
            c_name = c_fname

            # buf is C-contiguous, so every buf[i] is contiguous: the address can be used directly
            mv = buf[i]
            data_ptr = &mv[0, 0, 0]

            errbuf[0] = 0
            with nogil:
                rc = wi_write_png(c_name, data_ptr, width, height, stride,
                                  compression_level, compress_strategy, disable_filters,
                                  errbuf, 256)

            if rc != 0:
                failures += 1
                msg_b = errbuf
                msg = f"{fname}: {msg_b.decode('utf-8', 'replace')}"
                if first_error is None:
                    first_error = msg
                logger.error(f"write_images: could not write {msg}")

        if failures:
            raise RuntimeError(f"{failures} image(s) could not be written, first error: {first_error}")

    finally:
        # drop all views of the shared memory, release the descriptors, then detach
        mv = None
        buf = None
        if desc is not None:
            try:
                desc[:, 0] = 0
            except Exception:
                pass
        desc_mv = None
        desc = None

        # close shared memory (do NOT unlink: the parent owns it)
        for segment in (shm, desc_shm):
            if segment is not None:
                try:
                    segment.close()
                except Exception:
                    pass

    return 0
