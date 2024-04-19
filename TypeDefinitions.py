from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from typing import TypedDict, NewType

import numpy.typing as npt

StateType = TypedDict('StateType', {'img': npt.NDArray, 'img_count': int})
ImgDescType = NewType('ImgDescType', {'number_of_data_bytes': int, 'img_count': int})

ProcessType = NewType('ProcessType', {'process': Process, 'shm': SharedMemory})