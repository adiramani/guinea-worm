from .tools import process_data
from .model.population import HostPopulation, SinkPopulation
from .model.model import Model

class GuineaWormModel:
    model: Model
    

    def __init__(self, sink_info: list[dict], host_info: list[dict], model_info: dict):
        sink_pops = {}
        for sink_params in sink_info:
            sink_params["r0_worm_to_sink"] = (model_info["r0"] ** (model_info["transmission_asymmetry"]))
            tmp_sink = SinkPopulation(**sink_params)
            sink_pops[tmp_sink.population_name] = tmp_sink
        
        host_pops = {}
        for host_params in host_info:
            tmp_host = HostPopulation(**host_params)
            for sink_name in tmp_host.sink_name_order:
                sink_pops[sink_name].update_host_population(tmp_host.num_individuals)
            host_pops[tmp_host.population_name] = tmp_host
        
        population_info = {
            "host_populations": host_pops,
            "sink_populations": sink_pops
        }
        self.model = Model(**model_info, **population_info)

    def iterateFullModel(self):
        model_finished = False
        all_data = []
        while not(model_finished):
            model_finished, data_output = self.model.iterateModel()
            if data_output:
                all_data.append(data_output)
        return process_data(all_data)