import math
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
    start_infectivity: int
    r0_worm_to_sink: float
    num_emergences: int
    total_host_population: int

    def __init__(
        self,
        density: float, # copepods per liter
        size: float, # total liters
        population_name: str,
        r0_worm_to_sink: float,
        infectivity_rate: float = 0.0001,
        larval_death_rate: int = 30/360
    ):
        super().__init__(density * size, population_name, larval_death_rate)
        self.proportion_infected = infectivity_rate
        self.infective_larvae = math.floor(infectivity_rate * self.num_individuals)
        self.r0_worm_to_sink = r0_worm_to_sink
        self.num_emergences = 0
        self.total_host_population = 0
        self.mortality_rate = larval_death_rate

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

    def update_proportion_infected(self, timestep: int, NdNc: float):
        # rk4 for diferential equation
        new_proportion_infected = self.proportion_infected + (
            self.r0_worm_to_sink * 
            (self.num_emergences / self.total_host_population) * 
            NdNc *
            (1 - self.proportion_infected)
        ) - (
            self.mortality_rate * 
            self.proportion_infected *
            timestep
        )
        self.proportion_infected = min(max(new_proportion_infected, 0), 1)
        self.num_emergences = 0

    def get_proportion_infected(self):
        return self.proportion_infected
    
    def age(self, timestep: int, NdNc: float):
        self.update_proportion_infected(timestep, NdNc)

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
    ke: float
    # Dimensions: Rows are # of individuals columns are sinks, ordered by sink_name_order
    sink_interaction: list[list[int]]
    sink_name_order: list[str]

    def __init__(
        self,
        num_individuals: int,
        population_name: str,
        mortality_rate: float,
        initial_infected: int,
        worm_death_rate: float,
        worm_mating_probability: float,
        ke: float,
        sink_interaction_values: dict[str, dict[str, list[int]]],
        worm_maturity_age_days: int,
        max_worm_age: int,
    ):
        super().__init__(num_individuals, population_name, mortality_rate)
        self.worm_pop = Worms(
            worm_death_rate=worm_death_rate,
            individuals=num_individuals,
            mating_probability=worm_mating_probability,
            worm_maturity_age_days=worm_maturity_age_days,
            max_worm_age=max_worm_age
        )
        if (initial_infected > 0):
            self.worm_pop.male_worms[:initial_infected, 0] = 1
            self.worm_pop.female_worms[:initial_infected, 0] = 1
        self.ke = ke
        self.exposure_heterogeneity = np.random.gamma(
            shape=ke, scale=1 / ke, size=num_individuals
        )
        self.ages = np.full(num_individuals, 0)
        self.sink_name_order = list(sink_interaction_values.keys())
        self.sink_interaction = np.array(
            [sink_interaction_values[key]["interaction"] for key in self.sink_name_order]
        ).T

    def process_death(self, individuals: list[bool]):
        self.ages[individuals] = 0
        self.exposure_heterogeneity[individuals] = np.random.gamma(
            shape=self.ke, scale=1 / self.ke, size=sum(individuals)
        )
        self.worm_pop.process_host_death(individuals)

    def age(self, timestep: int):
        self.ages += timestep

        to_die = np.random.rand(len(self.ages)) < (1 - np.exp(-(self.mortality_rate) * self.ages))
        self.process_death(to_die)
        self.worm_pop.age(timestep)

    def worms_emerging(self, interaction_occured: list[bool]) -> float:
        return self.worm_pop.worms_emerging(interaction_occured)

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