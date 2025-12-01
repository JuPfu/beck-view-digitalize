# WriteImages.pxd
# Clean libpng declarations for WriteImages.pyx

from libc.stdint cimport uint32_t
from libc.stdio cimport FILE


# ----------------------------------------------------------------------
# Type definitions from png.h
# ----------------------------------------------------------------------
cdef extern from "png.h":

    ctypedef struct png_struct_def:
        pass

    ctypedef struct png_info_def:
        pass

    ctypedef png_struct_def* png_structp
    ctypedef png_info_def*  png_infop

    ctypedef png_struct_def** png_structpp
    ctypedef png_info_def**  png_infopp


# ----------------------------------------------------------------------
# Core creation / destruction
# ----------------------------------------------------------------------
cdef extern from "png.h":

    png_structp png_create_write_struct(
        const char* user_png_ver,
        void* error_ptr,
        void (*error_fn)(png_structp, const char*),
        void (*warn_fn)(png_structp, const char*)
    )

    png_infop png_create_info_struct(png_structp png_ptr)

    void png_destroy_write_struct(
        png_structpp png_ptr_ptr,
        png_infopp  info_ptr_ptr
    )


# ----------------------------------------------------------------------
# IO setup
# ----------------------------------------------------------------------
cdef extern from "png.h":

    void png_init_io(png_structp png_ptr, FILE* fp)


# ----------------------------------------------------------------------
# Image header setup
# ----------------------------------------------------------------------
cdef extern from "png.h":

    void png_set_IHDR(
        png_structp png_ptr,
        png_infop   info_ptr,
        uint32_t    width,
        uint32_t    height,
        int         bit_depth,
        int         color_type,
        int         interlace_method,
        int         compression_method,
        int         filter_method
    )

    void png_set_compression_level(png_structp png_ptr, int level)


# ----------------------------------------------------------------------
# Optional transforms (we use only png_set_bgr)
# ----------------------------------------------------------------------
cdef extern from "png.h":

    void png_set_bgr(png_structp png_ptr)


# ----------------------------------------------------------------------
# Writing operations
# ----------------------------------------------------------------------
cdef extern from "png.h":

    void png_write_info(png_structp png_ptr, png_infop info_ptr)

    void png_write_row(png_structp png_ptr, unsigned char* row)

    void png_write_image(
        png_structp png_ptr,
        unsigned char** row_pointers
    )

    void png_write_end(png_structp png_ptr, png_infop info_ptr)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
cdef extern from "png.h":
    int PNG_COLOR_TYPE_RGB
