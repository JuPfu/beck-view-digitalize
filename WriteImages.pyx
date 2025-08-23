import cython

import logging
import sys
from multiprocessing import shared_memory
from multiprocessing.resource_tracker import unregister

import cv2

# Use memory views for better performance than NumPy arrays
import numpy as np
cimport numpy as np

from TypeDefinitions import ImgDescType
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

def write_images(shm_name: cython.str,
                 img_desc: cython.list[ImgDescType],
                 img_width: cython.int,
                 img_height: cython.int,
                 output_path: Path) -> cython.list[ImgDescType]:
    """
    Write batch of images to persistent storage from shared memory.

    Args:
        shm_name: str -- Reference to shared memory.
        img_desc: list[ImgDescType] -- The size of the image data and frame number for each image in the chunk.
        img_width: int -- Width of images.
        img_height: int -- Height of images.
        output_path: Path -- Directory path to save images.

    Returns:
        list[ImgDescType]: The image descriptions array.
    """

    from concurrent.futures import ThreadPoolExecutor, as_completed

    cdef int total_size, start, end, frame_bytes, frame_count, success

    # Initialize shared memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
    except Exception as e:
        logger.error(f"Failed to initialize SharedMemory: {e}")
        return img_desc

    try:
        # Calculate total size of the image data
        total_size = sum(desc[0] for desc in img_desc)

        # Create a NumPy array view of the shared memory buffer
        data = np.ndarray((total_size,), dtype=np.uint8, buffer=shm.buf)

        # try:
        #    print(f"vor unregister {shm.name=}")
        #    unregister(shm.name, 'shared_memory')
        # except Exception:
        #    # already unregistered or not tracked — ignore
        #    print(f"already unregistered {shm.name=}")
        #    pass

        def write_single_image(start: int, end: int, frame_bytes: int, frame_count: int, suffix: str):
            try:
                # Set the output filename for the current frame
                filename = output_path / f"frame{frame_count:05d}{suffix}.png"
                # Reshape the slice of data to the image shape (height, width, 3)
                image_data = data[start:end].reshape((img_height, img_width, 3))
                # Write the image data to persistent storage
                success = cv2.imwrite(str(filename), image_data, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if not success:
                    logger.error(f"Failed to write image: {filename}")
            except Exception as e:
                logger.error(f"Error writing image frame {frame_count}: {e}")

        futures = []
        end = 0
        with ThreadPoolExecutor(max_workers=min(8, len(img_desc))) as executor:
            for frame_bytes, frame_count, suffix in img_desc:
                start = end
                end += frame_bytes  # Calculate the end index for the current image slice
                futures.append(executor.submit(write_single_image, start, end, frame_bytes, frame_count, suffix))

            for future in as_completed(futures):
                future.result()

    except Exception as e:
        logger.error(f"Error in child process: {e}")

    finally:
        try:
            shm.close()
        except Exception as e:
            logger.warning(f"SharedMemory close failed: {e}")

    return img_desc
