# WriteImages.pxd
# libpng declarations needed by WriteImages.pyx

from libc.stdint cimport uint32_t, uint8_t
from libc.stdio cimport FILE
from libc.setjmp cimport jmp_buf

cdef extern from "png.h":
    ctypedef struct png_struct_def:
        pass
    ctypedef struct png_info_def:
        pass

    ctypedef png_struct_def* png_structp
    ctypedef png_info_def* png_infop

    png_structp png_create_write_struct(
        const char* user_png_ver,
        void* error_ptr,
        void (*error_fn)(png_structp, const char*),
        void (*warn_fn)(png_structp, const char*)
    )

    png_infop png_create_info_struct(png_structp png_ptr)

    void png_destroy_write_struct(png_structp* png_ptr_ptr,
                                  png_infop* info_ptr_ptr)

    void png_init_io(png_structp png_ptr, FILE* fp)

    void png_set_IHDR(
        png_structp png_ptr,
        png_infop info_ptr,
        uint32_t width,
        uint32_t height,
        int bit_depth,
        int color_type,
        int interlace_method,
        int compression_method,
        int filter_method
    )

    void png_set_compression_level(png_structp png_ptr, int level)
    void png_set_compression_strategy(png_structp png_ptr, int strategy)
    void png_set_filter(png_structp png_ptr, int method, int filters)

    void png_set_bgr(png_structp png_ptr)

    void png_write_info(png_structp png_ptr, png_infop info_ptr)
    void png_write_row(png_structp png_ptr, unsigned char* row)
    void png_write_image(png_structp png_ptr, unsigned char** row_pointers)
    void png_write_end(png_structp png_ptr, png_infop info_ptr)

    # setjmp/longjmp helpers
    jmp_buf* png_jmpbuf(png_structp png_ptr)

# constants from png.h (we only import what's needed)
cdef extern from "png.h":
    int PNG_COLOR_TYPE_RGB
    int PNG_COLOR_TYPE_RGBA
    int PNG_ALL_FILTERS
    int PNG_FILTER_NONE
