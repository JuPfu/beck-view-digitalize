from multiprocessing.pool import AsyncResult
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import numpy.typing as npt

# Define the type of RGB image array with shape (height, width, 3)
RGBImageArray = npt.NDArray[np.uint8]

# Define StateType as a tuple containing an RGB image array and an integer (frame count)
StateType = tuple[RGBImageArray, int]

# Define SubjectDescType as a tuple containing an integer and a float (frame  count, time stamp)
SubjectDescType = tuple[int, float]

# Define ImgDescType as a tuple containing two integers (size of the image data in bytes, frame count)
ImgDescType = tuple[int, int]

# Define ProcessType as a tuple containing an AsyncResult and a SharedMemory object
ProcessType = tuple[AsyncResult, SharedMemory]
