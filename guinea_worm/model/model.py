from .intervention import Intervention, InterventionEvent
from .population import HostPopulation, SinkPopulation
import numpy as np
import scipy.stats as stats


class Model:
    time: int
    timestep: int
    endtime: int
    _days_in_year: int = 360
    _worm_rates: list[float] = []
    host_populations: dict[str, HostPopulation]
    sink_populations: dict[str, SinkPopulation]
    interventions: dict[InterventionEvent, Intervention]
    r0: float
    transmission_asymmetry: float
    verbose: bool
    emergence_events: dict[str, dict[str, int]]
    burnin_time: int
    NhNc: float
    NcNh: float

    def __init__(
        self,
        time: int,
        timestep: int,
        endtime: int,
        r0: float,
        r0_sink_to_worm: float,
        NcNh: float,
        burnin_time_years: int,
        host_populations: dict[str, HostPopulation],
        sink_populations: dict[str, SinkPopulation],
        interventions: dict[InterventionEvent, Intervention] = None,
        verbose: bool = False,
    ):
        self.time = time
        self.timestep = timestep
        self.endtime = endtime
        self.r0 = r0
        self.r0_sink_to_worm = r0_sink_to_worm
        self.NcNh = NcNh
        self.NhNc = 1 / NcNh
        self.burnin_time = burnin_time_years * self._days_in_year
        self.host_populations = host_populations
        self.sink_populations = sink_populations
        if interventions is not None:
            self.interventions = interventions
        self.verbose = verbose
        self.emergence_events = {}

    def get_seasonality_factor(self):
        if self.time < (self.burnin_time):
            return 1
        day_remainder = (self.time % self._days_in_year)
        if (day_remainder == 0 & self.time != 0):
            day_remainder = 360
        elif (self.time == 0):
            day_remainder == 1
        day_value = day_remainder / self._days_in_year
        return stats.beta.pdf(
            x=day_value,
            a=0.65 * (12.32 - 2) + 1,
            b=(1 - 0.65) * (12.32 - 2) + 1
        )

    def check_for_exposure_event(self):
        for host_population_name in self.host_populations:
            host_population = self.host_populations[host_population_name]
            n = host_population.worm_pop.worm_death_gamma_shape
            for index, sink_name in enumerate(host_population.sink_name_order):
                interactions = host_population.sink_interaction[:, index]
                interaction_occurred = np.random.rand(len(interactions)) <= interactions

                # Infection Event
                rate_of_infection_in = np.where(
                    interaction_occurred,
                    host_population.exposure_heterogeneity *
                    self.r0_sink_to_worm *
                    self.sink_populations[sink_name].larvae_injested(interaction_occurred) *
                    self.abate_efficacy() *
                    (
                        (self.sink_populations[sink_name].mortality_rate) * 
                        (
                            ((host_population.mortality_rate) + (n * host_population.worm_pop.worm_death_rate)) / 
                            (n * host_population.worm_pop.worm_death_rate)
                        ) ** n
                    ) *
                    self.NcNh * (self.get_seasonality_factor()) /
                    host_population.worm_pop.sex_ratio,
                    0.0
                ) * self.timestep

                
                new_worms_in = np.random.poisson(
                    lam=rate_of_infection_in
                )

                self._worm_rates.append(np.mean(new_worms_in))
                host_population.worm_pop.new_worms_injested(
                    new_worms_in
                )

                # Emergance Event
                if (host_population_name not in self.emergence_events) or (sink_name not in self.emergence_events[host_population_name]):
                    self.emergence_events[host_population_name] = {sink_name: {"worms": 0, "hosts": 0, "sum_worm_age": 0}}
                num_worms_emerging, num_hosts_with_emergence, emerging_worm_age = host_population.worms_emerging(
                    interaction_occurred,
                    teathering_efficacy=self.teathering_efficacy()
                )
                self.sink_populations[sink_name].add_infectivity_boost(num_worms_emerging)
                self.emergence_events[host_population_name][sink_name]["worms"] += num_worms_emerging
                self.emergence_events[host_population_name][sink_name]["hosts"] += num_hosts_with_emergence
                self.emergence_events[host_population_name][sink_name]["sum_worm_age"] += emerging_worm_age
                

    def _get_intervention_value(self, intervention_name, default_val):
        for event in self.interventions.keys():
            if self.time > event[0] and self.time < event[1] and event[2] == intervention_name:
                return self.interventions[event].intervention_value
        return default_val

    def teathering_efficacy(self) -> float:
        return self._get_intervention_value("teathering", 0)
    
    def abate_efficacy(self) -> float:
        return self._get_intervention_value("abate", 1)

    def iterateModel(self, output_data=False):
        if self.time > self.endtime:
            if self.verbose:
                print(f"Finished. Time: {self.time} days ({self.time / self._days_in_year} years)")
            return True, {
                "year": (self.time / self._days_in_year),
                "stats": self.printPopulationStats(self.verbose)
            }

        self.check_for_exposure_event()
        for population_name in self.host_populations:
            population = self.host_populations[population_name]
            population.age(timestep=self.timestep, current_time=self.time, burnin_time=self.burnin_time)

        for population_name in self.sink_populations:
            population = self.sink_populations[population_name]
            population.age(timestep=self.timestep, NhNc=self.NhNc, n=37) # TODO: Fix hard coding of n, possible have it built into emerging worms

        return_stats = {}
        print_summary = False
        if self.time % self._days_in_year == 0:
            if self.verbose:
                print(f"Starting iteration for year {self.time / self._days_in_year}")
                print_summary = True
        if output_data:
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
            num_infected_host = np.sum(host_population.worm_pop.get_total_worms() > 0)
            population_stats[host_population_name] = host_population.stats(verbose=print_summary)
            population_stats[host_population_name]["mean_worm_rate"] = np.mean(self._worm_rates)
            population_stats[host_population_name]["mating_prob"] = host_population.worm_pop.get_mating_probability()
            population_stats[host_population_name]["mean_age"] = np.mean(host_population.ages)
            population_stats[host_population_name]["num_infected_host"] = num_infected_host
            self._worm_rates = []

            for sink_name, value in self.emergence_events[host_population_name].items():
                population_stats[host_population_name][f"emergence_{sink_name}"] = value["worms"]
                population_stats[host_population_name][f"emergence_hosts_{sink_name}"] = value["hosts"]
                population_stats[host_population_name][f"emergences_per_host_{sink_name}"] = value["worms"] / host_population.num_individuals# if value["hosts"] != 0 else 0
                population_stats[host_population_name][f"emergent_host_prevalence_{sink_name}"] = value["hosts"] / host_population.num_individuals
                population_stats[host_population_name][f"emergences_per_infected_host_{sink_name}"] = value["worms"] / value["hosts"] if value["hosts"] != 0 else 0
                population_stats[host_population_name][f"emergence_average_worm_age_{sink_name}"] = value["sum_worm_age"] / value["worms"] if value["worms"] > 0 else 0
                self.emergence_events[host_population_name][sink_name]["worms"] = 0
                self.emergence_events[host_population_name][sink_name]["hosts"] = 0
                self.emergence_events[host_population_name][sink_name]["sum_worm_age"] = 0

        for sink_name in self.sink_populations:
            population_stats[sink_name] = self.sink_populations[sink_name].stats(verbose=print_summary)
        return population_stats

    def setDaysInYear(self, days: int) -> None:
        self._days_in_year = days
