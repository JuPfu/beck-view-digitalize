# TimingResult.pyx
import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef class TimingResult:
    def __init__(self, int max_frames):
        self.max_frames = max_frames
        self.size = 0

        # allocate NumPy array backing storage
        arr = np.zeros((max_frames, 7), dtype=np.float64)

        # create memoryview
        self.buf = arr

    cpdef void append(
            self,
            double count,
            double cycle,
            double work,
            double read,
            double latency,
            double wait_time,
            double total_work,
        ):
        cdef int i = self.size
        if i >= self.max_frames:
            return  # ignore overflow

        self.buf[i, 0] = count
        self.buf[i, 1] = cycle
        self.buf[i, 2] = work
        # self.buf[i, 3] = read # is assigend separately in DigitizeVideo.pyx
        self.buf[i, 4] = latency
        self.buf[i, 5] = wait_time
        self.buf[i, 6] = total_work
        self.size += 1

    #
    # Provide a Python-accessible property for DigitizeVideo
    #
    def to_numpy(self):
        """Return only the valid rows as NumPy array (copy!)."""
        import numpy as np
        return np.asarray(self.buf[:self.size, :])
