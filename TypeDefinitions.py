from typing import TypedDict

import numpy.typing as npt

StateType = TypedDict('StateType', {'img': npt.NDArray, 'img_count': int})
ImgDescType = TypedDict('ImgDescType', {'number_of_data_bytes': int, 'img_count': int})
