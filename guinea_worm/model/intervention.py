from typing import Tuple
import numpy as np

InterventionEvent = Tuple[int, str]


class Intervention:
    start_time: int
    end_time: int
    intervention_interval: int
    intervention_name: str

    intervention_event_times: list[int]

    def __init__(
        self,
        intervention_name: str,
        start_time: int,
        end_time: int,
        intervention_interval=None,
        intervention_event_times: list[int] = [],
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.intervention_name = intervention_name

        if (intervention_interval == None) and (len(intervention_event_times) == 0):
            raise ValueError(
                "One of intervention_interval or intervention_event_times should be defined"
            )
        if not (intervention_interval is None):
            self.intervention_event_times = np.arange(
                start_time, end_time, intervention_interval
            )
        else:
            self.intervention_event_times = intervention_event_times
