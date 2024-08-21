from guinea_worm.population import HostPopulation, SinkPopulation
from guinea_worm.model import Model
import numpy as np

host_pops = {
    "Dogs": HostPopulation(
        num_individuals=1000,
        population_name="Dogs",
        k=0.3,
        larval_release_rate=0.05,
        larvae_per_female_worm=5,
        sink_interaction_values={"copepod": np.full(1000, 0.5)},
    )
}
sink_pops = {
    "copepod": SinkPopulation(
        num_individuals=30000, population_name="copepod", infectivity_rate=0.0001
    )
}

model = Model(
    time=0,
    timestep=30,
    endtime=365 * 10,
    populations=host_pops,
    sink_populations=sink_pops,
)

model_finished = False
while not (model_finished):
    model_finished = model.iterateModel()
