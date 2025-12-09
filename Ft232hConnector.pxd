# Ft232hConnector.pxd
# Declarations for Ft232hConnector so the implementation (.pyx) can
# define the cdef class with the listed c-level attributes.

from TimingResult cimport TimingResult as CTimingResult

# C-level module-global (the C symbol must exist in .pyx too)
cdef CTimingResult timing_view

cdef class Ft232hConnector:
    """
    Public signature for the cdef class implemented in Ft232hConnector.pyx.
    Keep cdef attributes here so the implementation may define them.
    """

    # C attributes (must match those in the .pyx)
    cdef object logger

    cdef double LATENCY_THRESHOLD
    cdef int INITIAL_COUNT
    cdef bint gui
    cdef CTimingResult timing_view

    cdef object _ftdi
    cdef object _gpio
    cdef object signal_subject

    cdef int _OK1_mask
    cdef int _END_OF_FILM_mask

    cdef object _stop_event
    cdef object _thread
    cdef bint running

    cdef int max_count
    cdef object _timing_lock


# Accessor exported from module
cpdef CTimingResult get_timing_view()
cpdef object get_timing()
