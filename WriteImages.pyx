# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False, infer_types=True
import cython
import logging
import sys
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
# Use memory views for better performance than NumPy arrays
cimport numpy as np

from TypeDefinitions import ImgDescType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def write_images(buffer_name: cython.str,
                 img_desc: cython.list[ImgDescType],
                 img_width: cython.int,
                 img_height: cython.int,
                 output_path: Path) -> cython.list[ImgDescType]:
    """
    Write batch of images to persistent storage from shared memory.

    Args:
        buffer_name: str -- Reference to shared memory.
        img_desc: list[ImgDescType] -- The size of the image data and frame number for each image in the chunk.
        img_width: int -- Width of images.
        img_height: int -- Height of images.
        output_path: Path -- Directory path to save images.

    Returns:
        list[ImgDescType]: The image descriptions array.
    """

    cdef int total_size, start, end, frame_bytes, frame_count, success

    # Initialize shared memory
    try:
        shm = shared_memory.SharedMemory(name=buffer_name, create=False)
    except Exception as e:
        logger.error(f"Failed to initialize SharedMemory: {e}")
        return img_desc

    try:
        # Calculate total size of the image data
        total_size = len(img_desc) * img_height * img_width * 3

        # Create a NumPy array view of the shared memory buffer
        data: npt.NDArray[np.uint8] = np.ndarray((total_size,), dtype=np.uint8, buffer=shm.buf)

        end = 0

        # Iterate through image descriptions and write each image to persistent storage
        for frame_bytes, frame_count in img_desc:
            start = end
            end += frame_bytes  # Calculate the end index for the current image slice

            # Set the output filename for the current frame
            filename = output_path / f"frame{frame_count}.png"

            # Reshape the slice of data to the image shape (height, width, 3)
            image_data = data[start:end].reshape((img_height, img_width, 3))

            # Write the image data to persistent storage
            success = cv2.imwrite(str(filename), image_data)

            if not success:
                logger.error(f"Failed to write image: {filename}")

    except Exception as e:
        logger.error(f"Error in child process: {e}")

    finally:
        if shm is not None:
            shm.unlink()

    return img_desc
