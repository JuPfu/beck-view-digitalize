from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from typing import TypedDict

import numpy.typing as npt

StateType = TypedDict('StateType', {'img': npt.NDArray, 'img_count': int})
ImgDescType = TypedDict('ImgDescType', {'number_of_data_bytes': int, 'img_count': int})

ProcessDict = TypedDict('ProcessDict', {'process': Process, 'shm': SharedMemory})