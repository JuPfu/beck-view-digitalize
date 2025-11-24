# TimingResult.pxd

cdef class TimingResult:
    cdef:
        double[:, :] buf     # memoryview: (max_frames, 7)
        int max_frames
        int size              # number of written rows

        # count         frame count
        # cycle         time for a complete cycle including wait time
        # work          time from passing event to DigitizeView and returning from there
        # read          read time used by openCV
        # latency       wait time for OK1 pin reset back to low
        # wait_time     idle time until next OK1 signal
        # total work    cycle time without wait time
        #
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
