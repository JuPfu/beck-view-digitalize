import numpy.typing as npt
from mypy.typeshed.stdlib.multiprocessing.pool import AsyncResult
from mypy.typeshed.stdlib.multiprocessing.shared_memory import SharedMemory

StateType = tuple[npt.NDArray, int]
ImgDescType = tuple[int, int]

ProcessType = tuple[AsyncResult, SharedMemory]
