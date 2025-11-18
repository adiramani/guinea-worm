from typing import Tuple
import numpy as np

InterventionEvent = Tuple[int, int, str]


class Intervention:
    start_time: int
    end_time: int
    intervention_interval: int
    intervention_name: str
    intervention_value: float
    months_to_apply: list[int]

    intervention_event_times: list[int]

    def __init__(
        self,
        intervention_name: str,
        start_time: int,
        end_time: int,
        intervention_value: float,
        intervention_interval=None,
        intervention_event_times: list[int] = [],
        months_to_apply: list[int] = []
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.intervention_name = intervention_name
        self.intervention_value = intervention_value
        self.months_to_apply = months_to_apply

        # if intervention_interval is not None:
        #     self.intervention_event_times = np.arange(
        #         start_time, end_time, intervention_interval
        #     )
        # else:
        #     self.intervention_event_times = intervention_event_times
