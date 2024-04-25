from multiprocessing.pool import AsyncResult
from multiprocessing.shared_memory import SharedMemory

import numpy.typing as npt


StateType = tuple[npt.NDArray, int]
ImgDescType = tuple[int, int]

ProcessType = tuple[AsyncResult, SharedMemory]
