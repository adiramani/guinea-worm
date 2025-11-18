import math

from guinea_worm.model.params import HostParams, SinkParams
from .worms import Worms
import numpy as np
import random


class Population:
    num_individuals: int
    population_name: str
    mortality_rate: float

    def __init__(self, num_individuals: int, population_name: str, mortality_rate: float):
        self.num_individuals = num_individuals
        self.population_name = population_name
        self.mortality_rate = mortality_rate


class SinkPopulation(Population):
    larvae_injestion_rate: float
    proportion_infected: float
    num_emergences: int
    total_host_population: int

    def __init__(
        self,
        params: SinkParams,
        # density: float, # copepods per liter
        # size: float, # total liters
        # population_name: str,
        # infectivity_rate: float,
        # larval_death_rate: int,
    ):
        super().__init__(SinkParams.density * SinkParams.size, SinkParams.population_name, SinkParams.larval_death_rate)
        self.proportion_infected = SinkParams.infectivity_rate
        self.num_emergences = 0
        self.total_host_population = 0
        self.mortality_rate = SinkParams.larval_death_rate

    def update_host_population(self, num_individuals: int):
        self.total_host_population += num_individuals

    def larvae_injested(self, infection_interaction: list[bool]) -> list[int]:        
        infected_sinks_injested_indiv = np.full(
            len(infection_interaction), 
            self.get_proportion_infected()
        )
        return infected_sinks_injested_indiv
    
    def add_infectivity_boost(self, num_emergences: float):
        self.num_emergences += num_emergences

    def update_proportion_infected(self, timestep: int, a: float):
        new_proportion_infected = self.proportion_infected + (
            a *
            self.total_host_population *
            (self.num_emergences / self.total_host_population) * #TODO: simplify to just num_emergences
            (1 - self.proportion_infected)
        ) - (
            self.mortality_rate * 
            self.proportion_infected *
            timestep
        )
        new_proportion_infected = (self.num_emergences * a) / ((self.mortality_rate * timestep) + (self.num_emergences * a))
        self.proportion_infected = min(max(new_proportion_infected, 0), 1)
        self.num_emergences = 0

    def get_proportion_infected(self):
        return self.proportion_infected
    
    def age(self, timestep: int, a: float):
        self.update_proportion_infected(timestep, a)

    def stats(self, verbose=False):        
        if(verbose):
            print(
                f"{self.population_name} Infection prevalence: {self.get_proportion_infected()}"
            )
        return {
            "infective_larvae": self.get_proportion_infected()
        }
            

class HostPopulation(Population):
    ages: list[int]
    worm_pop: Worms
    exposure_heterogeneity: list[int]
    k: float
    # Dimensions: Rows are # of individuals columns are sinks, ordered by sink_name_order
    sink_interaction: list[list[int]]
    sink_name_order: list[str]

    def __init__(
        self,
        params: HostParams,
    ):
        super().__init__(params.num_individuals, params.population_name, params.mortality_rate)
        self.worm_pop = Worms(
            timestep=params.timestep,
            worm_death_rate=params.worm_death_rate,
            individuals=params.num_individuals,
            mating_probability=params.worm_mating_probability,
            worm_maturity_age_days=params.worm_maturity_age_days,
            max_worm_age=params.max_worm_age,
            worm_death_gamma_shape=params.worm_death_gamma_shape,
        )
        if (params.initial_infected > 0):
            self.worm_pop.male_worms[:params.initial_infected, 0] = params.worms_to_infect_with
            self.worm_pop.female_worms[:params.initial_infected, 0] = params.worms_to_infect_with

        self.k = params.k
        self.exposure_heterogeneity = np.random.gamma(
            shape=params.k, scale=1 / params.k, size=params.num_individuals
        )
        self.ages = np.random.randint(0, 901, size=params.num_individuals)
        self.sink_name_order = list(params.sink_interaction_values.keys())
        self.sink_interaction = np.array(
            [params.sink_interaction_values[key]["interaction"] for key in self.sink_name_order]
        ).T

    def process_death(self, individuals: list[bool]):
        self.ages[individuals] = 0
        self.exposure_heterogeneity[individuals] = np.random.gamma(
            shape=self.k, scale=1 / self.k, size=sum(individuals)
        )
        self.worm_pop.process_host_death(individuals)

    def age(self, timestep: int, current_time: int, burnin_time: int):
        self.ages += timestep

        to_die = np.random.rand(len(self.ages)) < np.full(len(self.ages), (self.mortality_rate * timestep))#(1 - np.exp(-(self.mortality_rate) * self.ages))
        self.worm_pop.age(timestep, current_time, burnin_time)
        self.process_death(to_die)

    def worms_emerging(self, interaction_occured: list[bool], tethering_efficacy: float) -> tuple[float, float, list[float], float]:
        return self.worm_pop.worms_emerging(interaction_occured, tethering_efficacy)

    def stats(self, verbose=False) -> dict[str, int]:
        total_worm_burden = self.worm_pop.get_total_worms()
        num_infected_with_worm = np.mean(
            np.array(total_worm_burden) > 0
        )

        worm_load_per_person =  np.mean(total_worm_burden)
        female_worm_burden = np.array(self.worm_pop.get_female_worm_burden())
        female_worm_load_per_person = np.mean(female_worm_burden)

        female_worm_prev = np.sum(female_worm_burden > 0) / len(female_worm_burden)
        
        if(verbose):
            print(
                f"Worm Infection prevalence: {num_infected_with_worm}"
            )

            print(
                f"Worms Load per Person: {worm_load_per_person}\nFemale Worm Load per Person: {female_worm_load_per_person}"
            )

        return {
            "total_worm_prev": num_infected_with_worm,
            "female_worm_prev": female_worm_prev,
            "total_worm_load_per_person": worm_load_per_person,
            "female_worm_load_per_person": female_worm_load_per_person
        }