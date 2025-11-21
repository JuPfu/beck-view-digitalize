# WriteImages.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

from multiprocessing import shared_memory
from pathlib import Path

cimport numpy as cnp
import numpy as np
import cv2

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


# typed signature — usable from apply_async in the process pool
cpdef void write_images(
        str shm_name,
        str desc_name,
        int frames_total,
        int img_width,
        int img_height,
        object output_path):
    """
    Read frames and per-frame descriptors from shared memory and write PNG files.

    Expected SHM layout (per-DigitizeVideo.pyx):
      - pixel SHM: flat uint8 array sized frames_total * img_bytes
      - desc SHM:  uint32 array shaped (frames_total, 3) with columns:
            [0] = img_bytes (uint32)
            [1] = frame_count (uint32)
            [2] = bracket_index (uint32)  # 0->'a',1->'b',2->'c'
    """

    cdef:
        cnp.ndarray[cnp.uint8_t, ndim=1] pixel_np
        cnp.ndarray[cnp.uint32_t, ndim=2] desc_np
        int i
        int img_bytes = img_width * img_height * 3
        int offset
        int start, end
        int stored_bytes
        int frame_count
        int bracket_index
        object out_path_obj
        object fname
        tuple suffix_map = ('a', 'b', 'c')

    # normalize output path (accepts str or Path)
    try:
        out_path_obj = Path(output_path)
    except Exception:
        out_path_obj = Path(str(output_path))

    # attach SHM objects (python-level)
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
    except Exception as e:
        # can't attach pixel SHM
        logger.error(f"[write_images] failed to attach pixel SHM '{shm_name}': {e}")
        return

    try:
        desc_shm = shared_memory.SharedMemory(name=desc_name, create=False)
    except Exception as e:
        logger.error(f"[write_images] failed to attach desc SHM '{desc_name}': {e}")
        try:
            shm.close()
        except Exception:
            pass
        return

    try:
        # Map pixel SHM -> 1D uint8 ndarray
        pixel_np = np.ndarray((frames_total * img_bytes,), dtype=np.uint8, buffer=shm.buf)

        # Map descriptor SHM -> (frames_total, 3) uint32 ndarray
        desc_np = np.ndarray((frames_total, 3), dtype=np.uint32, buffer=desc_shm.buf)
    except Exception as e:
        logger.error(f"[write_images] failed to map SHM into numpy arrays: {e}")
        try:
            shm.close()
        except Exception:
            pass
        try:
            desc_shm.close()
        except Exception:
            pass
        return

    # iterate frames (frames_total slots). For each slot compute offset = i * img_bytes
    for i in range(frames_total):
        # read per-frame descriptor (as integers)
        # desc_np[i,0] is the stored img_bytes (uint32)
        stored_bytes = int(desc_np[i, 0])
        frame_count = int(desc_np[i, 1])
        bracket_index = int(desc_np[i, 2])

        # if stored_bytes is zero or mismatches expected size, skip
        if stored_bytes == 0:
            # nothing stored in this slot
            continue

        if stored_bytes != img_bytes:
            # mismatch: warn and continue
            logger.error(f"[write_images] size mismatch slot={i} stored={stored_bytes} expected={img_bytes}")
            # still attempt to read min(stored_bytes, img_bytes)
            # but to keep simple, we'll skip
            continue

        offset = i * img_bytes
        start = offset
        end = offset + img_bytes

        try:
            # zero-copy slice view into shared memory
            frame_flat = pixel_np[start:end]
            # reshape to (h, w, 3)
            frame_arr = frame_flat.reshape((img_height, img_width, 3))

            # suffix from bracket_index (fallback to 'a'.. if out of range)
            if 0 <= bracket_index < len(suffix_map):
                suffix = suffix_map[bracket_index]
            else:
                suffix = 'a'

            fname = out_path_obj / f"frame{frame_count:05d}{suffix}.png"

            # write with OpenCV (cv2.imwrite returns True/False)
            success = cv2.imwrite(str(fname), frame_arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not success:
                logger.error(f"[write_images] failed to write {fname}")
        except Exception as e:
            logger.error(f"[write_images] error writing frame slot={i} frame={frame_count}: {e}")

    # cleanup
    try:
        shm.close()
    except Exception:
        pass
    try:
        desc_shm.close()
    except Exception:
        pass

    return
