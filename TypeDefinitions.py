from typing import TypedDict

import numpy.typing as npt

StateType = TypedDict('StateType', {'img': npt.NDArray, 'img_count': int})
