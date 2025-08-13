# cython: language_level=3
# cython.infer_types(True)
import cython
from multiprocessing.pool import AsyncResult
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt

# Define the type of RGB image array with shape (height, width, 3)
RGBImageArray = npt.NDArray[np.uint8]

# Define SubjectDescType as a tuple containing an integer and a float (frame  count, time stamp)
SubjectDescType = tuple[int, float]

# Define ImgDescType as a tuple containing two integers (size of the image data in bytes, frame count, suffix string)
ImgDescType = tuple[int, int, str]

# Define ProcessType as a tuple containing an AsyncResult and a SharedMemory object
ProcessType = tuple[AsyncResult, SharedMemory]
