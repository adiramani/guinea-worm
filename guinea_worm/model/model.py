from .intervention import Intervention, InterventionEvent
from .population import HostPopulation, SinkPopulation
import numpy as np


class Model:
    time: int
    timestep: int
    endtime: int
    _days_in_year: int = 360
    host_populations: dict[str, HostPopulation]
    sink_populations: dict[str, SinkPopulation]
    interventions: dict[InterventionEvent, Intervention]
    r0: float
    transmission_asymmetry: float
    verbose: bool
    emergence_events: dict[str, dict[str, int]]
    NdNc: float
    NcNd: float

    def __init__(
        self,
        time: int,
        timestep: int,
        endtime: int,
        r0: float,
        transmission_asymmetry: float,
        NcNd: float,
        host_populations: dict[str, HostPopulation],
        sink_populations: dict[str, SinkPopulation],
        interventions: dict[InterventionEvent, Intervention] = None,
        verbose: bool = False,
    ):
        self.time = time
        self.timestep = timestep
        self.endtime = endtime
        self.r0 = r0
        self.transmission_asymmetry = transmission_asymmetry
        self.NcNd = NcNd
        self.NdNc = 1 / NcNd
        self.host_populations = host_populations
        self.sink_populations = sink_populations
        if interventions is not None:
            self.interventions = interventions
        self.verbose = verbose
        self.emergence_events = {}

    def check_for_exposure_event(self):
        for host_population_name in self.host_populations:
            host_population = self.host_populations[host_population_name]
            for index, sink_name in enumerate(host_population.sink_name_order):
                interactions = host_population.sink_interaction[:, index]
                interaction_occurred = np.random.rand(len(interactions)) < interactions

                # Infection Event
                rate_of_infection_in = np.where(
                    interaction_occurred,
                    #host_population.exposure_heterogeneity *
                    (self.r0 ** (1 - self.transmission_asymmetry)) *
                    self.sink_populations[sink_name].larvae_injested(interaction_occurred) *
                    (
                        self.sink_populations[sink_name].mortality_rate * 
                        (
                            (host_population.mortality_rate + host_population.worm_pop.worm_death_rate) / 
                            host_population.worm_pop.worm_death_rate)
                    ) *
                    self.NcNd *
                    self.timestep /
                    host_population.worm_pop.sex_ratio,
                    0.0
                )

                #rate_of_infection_in += np.sum(host_population.worm_pop.get_total_worms()) / host_population.num_individuals
                
                new_worms_in = np.random.poisson(
                    lam=rate_of_infection_in
                )

                # new_worms_in = 1 + (1 - rate_of_infection_in)
                host_population.worm_pop.new_worms_injested(
                    new_worms_in
                )

                # Emergance Event
                if (host_population_name not in self.emergence_events) or (sink_name not in self.emergence_events[host_population_name]):
                    self.emergence_events[host_population_name] = {sink_name: 0}
                num_worms_emerging = host_population.worms_emerging(
                    interaction_occurred
                )
                self.sink_populations[sink_name].add_infectivity_boost(num_worms_emerging)
                self.emergence_events[host_population_name][sink_name] += num_worms_emerging

    def iterateModel(self):
        if self.time > self.endtime:
            if self.verbose:
                print(f"Finished. Time: {self.time} days ({self.time / self._days_in_year} years)")
            return True, {
                "year": (self.time / self._days_in_year),
                "stats": self.printPopulationStats(self.verbose)
            }

        for population_name in self.host_populations:
            population = self.host_populations[population_name]
            population.age(timestep=self.timestep)

        for population_name in self.sink_populations:
            population = self.sink_populations[population_name]
            population.age(timestep=self.timestep, NdNc=self.NdNc)
        self.check_for_exposure_event()

        return_stats = {}
        print_summary = False
        if self.time % self._days_in_year == 0:
            if self.verbose:
                print(f"Starting iteration for year {self.time / self._days_in_year}")
                print_summary = True
        return_stats = {
            "year": (self.time / self._days_in_year),
            "stats": self.printPopulationStats(print_summary)
        }
        self.time += self.timestep
        return False, return_stats

    def printPopulationStats(self, print_summary: bool):
        population_stats = {}
        for host_population_name in self.host_populations:
            host_population = self.host_populations[host_population_name]
            population_stats[host_population_name] = host_population.stats(verbose=print_summary)

            for sink_name, value in self.emergence_events[host_population_name].items():
                population_stats[host_population_name][f"emergence_{sink_name}"] = value
                self.emergence_events[host_population_name][sink_name] = 0
                population_stats[host_population_name]["Re"] = (
                    (self.r0 ** (1 - self.transmission_asymmetry)) *
                    (self.r0 ** self.transmission_asymmetry) *
                    # proportion of all female worms that are classified as possibly "fertile"?
                    #host_population.worm_pop.get_mating_probability() *
                    (1-self.sink_populations[sink_name].get_proportion_infected())
                )

        for sink_name in self.sink_populations:
            population_stats[sink_name] = self.sink_populations[sink_name].stats(verbose=print_summary)
        return population_stats

    def setDaysInYear(self, days: int) -> None:
        self._days_in_year = days
