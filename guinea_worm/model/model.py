from guinea_worm.model.params import ModelParams
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
    verbose: bool
    emergence_events: dict[str, dict[str, int]]
    burnin_time: int
    a: float
    year_over_year_r0_factor: list[float]
    seasonality_start: int
    beta_dist_z: float
    beta_dist_q: float
    beta_dist_b: float

    def __init__(
        self,
        params: ModelParams
    ):
        self.time = params.time
        self.timestep = params.timestep
        self.endtime = params.endtime
        self.r0 = params.r0
        self.a = params.a
        self.beta_dist_z = params.beta_dist_z
        self.beta_dist_q = params.beta_dist_q
        self.beta_dist_b = params.beta_dist_b
        self.year_over_year_r0_factor = params.year_over_year_r0_factor
        self.seasonality_start = params.seasonality_start
        self.burnin_time = params.burnin_time_years * self._days_in_year
        self.host_populations = params.host_populations
        self.sink_populations = params.sink_populations
        if params.interventions is not None:
            self.interventions = params.interventions
        self.verbose = params.verbose
        self.emergence_events = {}

    def _get_model_month(self):
        day_of_year = self.time % self._days_in_year
        return (day_of_year // (self._days_in_year / 12)) + 1

    def get_seasonality_factor(self):
        return (
            self.get_year_over_year_seasonality_factor() * 
            self.get_within_year_seasonality_factor()
        )

    def get_year_over_year_seasonality_factor(self):
        year = self.time // self._days_in_year
        if year > self.seasonality_start:
            index = year - self.seasonality_start
            if index >= len(self.year_over_year_r0_factor):
                index = 0
            return self.year_over_year_r0_factor[index]
        return 1

    def get_within_year_seasonality_factor(self):
        year = self.time // self._days_in_year
        if year > self.seasonality_start:
            index = year - self.seasonality_start
            if index >= len(self.beta_dist_z):
                index = len(self.beta_dist_q)-1
        else:
            index = len(self.beta_dist_z)-1
        if self.time < (self.burnin_time):
            return 1
        day_remainder = (self.time % self._days_in_year)
        if (day_remainder == 0 & self.time != 0):
            day_remainder = 360
        elif (self.time == 0):
            day_remainder == 1
        day_value = day_remainder / self._days_in_year
        return (
            (1 - self.beta_dist_b[index]) *

            stats.beta.pdf(
                day_value,
                self.beta_dist_z[index] * (self.beta_dist_q[index] - 2) + 1,
                (1 - self.beta_dist_z[index]) * (self.beta_dist_q[index] - 2) + 1
            ) + 

            (self.beta_dist_b[index])
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
                    self.r0 *
                    self.sink_populations[sink_name].larvae_injested(interaction_occurred) *
                    (1 - self.abate_efficacy()) *
                    self.sink_populations[sink_name].mortality_rate *
                    (
                        (
                            ((host_population.mortality_rate) + (n * host_population.worm_pop.worm_death_rate)) / 
                            (n * host_population.worm_pop.worm_death_rate)
                        ) ** n
                    ) *
                    (self.get_seasonality_factor()) /
                    (
                        host_population.worm_pop.sex_ratio *
                        len(host_population.ages) *
                        self.a
                    ),
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
                    self.emergence_events[host_population_name] = {sink_name: {"worms": 0, "tethered_worms": 0, "hosts": 0, "sum_worm_age": 0, "host_ids": set(), "infectious_host_ids": set()}} #TODO: Need to track emergences per individual host
                num_infectious_worms_emerging, num_tethered_worms_emerging, num_hosts_with_emergence, indices_of_hosts_with_emergence, indices_of_hosts_with_infectious_emergence, emerging_worm_age = host_population.worms_emerging(
                    interaction_occurred,
                    tethering_efficacy=self.tethering_efficacy()
                )

                self.sink_populations[sink_name].add_infectivity_boost(num_infectious_worms_emerging)
                self.emergence_events[host_population_name][sink_name]["worms"] += num_infectious_worms_emerging
                self.emergence_events[host_population_name][sink_name]["tethered_worms"] += num_tethered_worms_emerging
                self.emergence_events[host_population_name][sink_name]["hosts"] += num_hosts_with_emergence
                self.emergence_events[host_population_name][sink_name]["sum_worm_age"] += emerging_worm_age
                self.emergence_events[host_population_name][sink_name]["host_ids"].update(set(indices_of_hosts_with_emergence))
                self.emergence_events[host_population_name][sink_name]["infectious_host_ids"].update(set(indices_of_hosts_with_infectious_emergence))
                

    def _get_intervention_value(self, intervention_name, default_val):
        for event in self.interventions.keys():
            if self.time > event[0] and self.time < event[1] and event[2] == intervention_name:
                intervention = self.interventions[event]
                if intervention.intervention_name == "abate":
                    if self._get_model_month() not in intervention.months_to_apply:
                        continue
                return intervention.intervention_value
        return default_val

    def tethering_efficacy(self) -> float:
        return self._get_intervention_value("tethering", 0)
    
    def abate_efficacy(self) -> float:
        return self._get_intervention_value("abate", 0)

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
            population.age(timestep=self.timestep, a = self.a)

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
            population_stats[host_population_name]["R0(t)"] = self.r0 * self.get_seasonality_factor()
            self._worm_rates = []

            for sink_name, value in self.emergence_events[host_population_name].items():
                total_worms = value["worms"] + value["tethered_worms"]
                population_stats[host_population_name][f"emergence_{sink_name}"] = total_worms
                population_stats[host_population_name][f"emergence_infectious_{sink_name}"] = value["worms"]
                population_stats[host_population_name][f"emergence_tethered_{sink_name}"] = value["tethered_worms"]
                population_stats[host_population_name][f"emergence_hosts_{sink_name}"] = value["hosts"]
                population_stats[host_population_name][f"emergent_host_cases_{sink_name}"] = len(value["host_ids"])
                population_stats[host_population_name][f"emergences_per_host_{sink_name}"] = total_worms / host_population.num_individuals# if value["hosts"] != 0 else 0
                population_stats[host_population_name][f"infectious_emergences_per_host_{sink_name}"] = value["worms"] / host_population.num_individuals# if value["hosts"] != 0 else 0
                population_stats[host_population_name][f"emergent_host_prevalence_{sink_name}"] = len(value["host_ids"]) / host_population.num_individuals
                population_stats[host_population_name][f"infectious_emergent_host_prevalence_{sink_name}"] = len(value["infectious_host_ids"]) / host_population.num_individuals
                population_stats[host_population_name][f"emergences_per_infected_host_{sink_name}"] = total_worms / len(value["host_ids"]) if len(value["host_ids"]) > 0 else np.nan
                population_stats[host_population_name][f"infectious_emergences_per_infected_host_{sink_name}"] = value["worms"] / len(value["infectious_host_ids"]) if len(value["infectious_host_ids"]) > 0 else np.nan
                population_stats[host_population_name][f"emergence_average_worm_age_{sink_name}"] = value["sum_worm_age"] / total_worms if value["worms"] > 0 else 0
                self.emergence_events[host_population_name][sink_name]["worms"] = 0
                self.emergence_events[host_population_name][sink_name]["tethered_worms"] = 0
                self.emergence_events[host_population_name][sink_name]["hosts"] = 0
                self.emergence_events[host_population_name][sink_name]["sum_worm_age"] = 0
                self.emergence_events[host_population_name][sink_name]["host_ids"] = set()
                self.emergence_events[host_population_name][sink_name]["infectious_host_ids"] = set()

        for sink_name in self.sink_populations:
            population_stats[sink_name] = self.sink_populations[sink_name].stats(verbose=print_summary)
        return population_stats

    def setDaysInYear(self, days: int) -> None:
        self._days_in_year = days
