# TimingResult.pxd

cdef class TimingResult:
    cdef:
        double[:, :] buf     # memoryview: (max_frames, 7)
        int max_frames
        int size              # number of written rows
        inline void append(
            self,
            double count,
            double cycle,
            double work,
            double read,
            double latency,
            double wait_time,
            double total_work,
        )
