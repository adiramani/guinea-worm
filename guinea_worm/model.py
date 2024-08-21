from guinea_worm.intervention import Intervention, InterventionEvent
from guinea_worm.population import HostPopulation, SinkPopulation
import numpy as np


class Model:
    time: int
    timestep: int
    endtime: int
    _days_in_year: int = 365
    host_populations: dict[str, HostPopulation]
    sink_populations: dict[str, SinkPopulation]
    interventions: dict[InterventionEvent, Intervention]

    foi_in: int = 1
    foi_out: int = 1

    def __init__(
        self,
        time: int,
        timestep: int,
        endtime: int,
        populations: dict[str, HostPopulation],
        sink_populations: dict[str, SinkPopulation],
        interventions: dict[InterventionEvent, Intervention] = None,
    ):
        self.time = time
        self.timestep = timestep
        self.endtime = endtime
        self.populations = populations
        self.sink_populations = sink_populations
        if not (interventions is None):
            self.interventions = interventions

    def check_for_exposure_event(self):
        larvae_into_sinks = {name: 0 for name in self.sink_populations.keys()}
        any_interaection_occurred = False
        for population_name in self.populations:
            population = self.populations[population_name]
            for index, value in enumerate(population.sink_name_order):
                interactions = population.sink_interaction[:, index]
                interaction_occurred = np.random.rand(len(interactions)) < interactions
                if interaction_occurred.any():
                    any_interaection_occurred = True

                # Infection Event
                rate_of_infection_in = np.where(
                    interaction_occurred,
                    population.exposure_heterogeneity
                    * self.foi_in
                    * self.sink_populations[value].getNumInfected(),
                    0,
                )
                population.worm_pop.larvae[:, 0] = np.random.poisson(
                    lam=rate_of_infection_in, size=population.num_individuals
                )

                # Emergance Event
                larvae_into_sinks[value] += population.check_emergences(
                    interaction_occurred
                )
        if any_interaection_occurred:
            print(f"Larvae into sink: {larvae_into_sinks}")
        return larvae_into_sinks

    def iterateModel(self):
        if self.time > self.endtime:
            return True

        for population_name in self.populations:
            population = self.populations[population_name]
            population.age(timestep=self.timestep)
        larvae_into_sinks = self.check_for_exposure_event()

        if self.time % self._days_in_year == 0:
            print(f"Starting iteration for year {self.time / self._days_in_year}")
            for population_name in self.populations:
                population = self.populations[population_name]
                population.stats()

        self.time += self.timestep
        return False

    def setDaysInYear(self, days: int) -> None:
        self._days_in_year = days
