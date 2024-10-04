# cython: language_level=3
# cython.infer_types(True)
import cython
timing: [{
    "count": cython.int,
    "cycle": cython.double,
    "work": cython.double,
    "read": cython.double,
    "latency": cython.double,
    "wait_time": cython.double,
    "total_work": cython.double
}] = []
