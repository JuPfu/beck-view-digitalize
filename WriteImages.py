# Set up logging configuration
import logging
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

from TypeDefinitions import ImgDescType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def write_images(buffer_name: str,
                 img_desc: [ImgDescType],
                 img_width: int,
                 img_height: int,
                 output_path: Path) -> ImgDescType:
    """
    Write batch of images to persistent storage.
    Images are delivered via shared memory.

    :parameter
        shared_memory_buffer_name: str -- Reference to shared memory

        img_desc: [ImgDescType] -- Array containing the data and number for each image of the chunk

        img_width: int -- Width of image

        img_height: int -- Height of image
    :returns
        ImgDescType
    """

    # get access to shared memory
    shm = shared_memory.SharedMemory(buffer_name)
    # number of images in shared buffer is deduced from length of img_desc
    # re-shape bytes from shared buffer into ndarray type with data type uint8
    data = np.ndarray((len(img_desc) * img_height * img_width * 3,), dtype=np.uint8, buffer=shm.buf)

    end: int = 0

    # write all images to persistent storage
    try:
        for (frame_bytes, frame_count) in img_desc:
            start = end
            end += frame_bytes  # add number of data bytes to designate end of current picture

            filename = output_path / f"frame{frame_count}.png"
            success: bool = cv2.imwrite(str(filename), data[start:end].reshape((img_height, img_width, 3)))
            if not success:
                logger.error(f"Could not write {filename=}")
    except Exception as e:
        logger.error(f"{e}")

    return img_desc
