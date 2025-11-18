from dataclasses import asdict, is_dataclass
from guinea_worm.model.params import HostParams, ModelParams, SinkParams
from .tools import process_data
from .model.population import HostPopulation, SinkPopulation
from .model.model import Model
from .model.intervention import Intervention
import numpy as np

class GuineaWormModel:
    model: Model
    

    def __init__(self, sink_info: list[dict], host_info: list[dict], model_info: dict, intervention_info: list[dict] = []):
        sink_pops = {}
        for sink_params in sink_info:
            tmp_sink = SinkPopulation(self.update_params(SinkParams, sink_params))#**sink_params)
            sink_pops[tmp_sink.population_name] = tmp_sink
        
        host_pops = {}
        for host_params in host_info:
            tmp_host = HostPopulation(self.update_params(HostParams, host_params))#**host_params)
            for sink_name in tmp_host.sink_name_order:
                sink_pops[sink_name].update_host_population(tmp_host.num_individuals)
            host_pops[tmp_host.population_name] = tmp_host

        interventions = {}
        for intervention_params in intervention_info:
            tmp_intervention = Intervention(**intervention_params)
            interventions[(
                tmp_intervention.start_time,
                tmp_intervention.end_time,
                tmp_intervention.intervention_name
            )] = tmp_intervention
        
        processed_intervention_info = {
            "interventions": interventions
        }
        population_info = {
            "host_populations": host_pops,
            "sink_populations": sink_pops
        }
        model_params = self.update_params(ModelParams, model_info, population_info, processed_intervention_info)
        self.model = Model(model_params)

    def update_params(self, base_params, *overrides):
        if not is_dataclass(base_params):
            raise TypeError(f"{base_params} must be a dataclass")
        
        base = asdict(base_params())

        for d in overrides:
            if d is not None:
                filtered_dict = {k: v for k, v in d.items() if v is not None}
                base.update(filtered_dict)

        return base_params(**base)

    def iterateFullModel(self, output_interval_years=1):
        output_times = np.arange(0, self.model.endtime, step=output_interval_years * self.model._days_in_year)
        model_finished = False
        all_data = []
        while not(model_finished):
            should_output = (self.model.time == output_times).any()
            model_finished, data_output = self.model.iterateModel(output_data=should_output)
            if data_output:
                all_data.append(data_output)
        return process_data(all_data)