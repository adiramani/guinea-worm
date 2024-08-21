import math
from guinea_worm.worms import Worms
import numpy as np


class Population:
    num_individuals: int
    population_name: str

    def __init__(self, num_individuals: int, population_name: str):
        self.num_individuals = num_individuals
        self.population_name = population_name


class SinkPopulation(Population):
    infectivity_rate: float

    def __init__(
        self,
        num_individuals: int,
        population_name: str,
        infectivity_rate: float = 0.0001,
    ):
        super().__init__(num_individuals, population_name)
        self.infectivity_rate = infectivity_rate

    def getNumInfected(self):
        return math.floor(self.num_individuals * self.infectivity_rate)


class HostPopulation(Population):
    ages: list[int]
    worm_pop: Worms
    k: float
    exposure_heterogeneity: list[int]
    # Dimensions: Rows are # of individuals columns are sinks, ordered by sink_name_order
    sink_interaction: list[list[int]]
    sink_name_order: list[str]

    def __init__(
        self,
        num_individuals: int,
        population_name: str,
        k: int,
        larval_release_rate: float,
        larvae_per_female_worm: float,
        sink_interaction_values: dict[str, list[int]],
    ):
        super().__init__(num_individuals, population_name)
        self.worm_pop = Worms(
            max_worm_age=365,
            max_larval_age=30,
            individuals=num_individuals,
            larval_release_rate=larval_release_rate,
            larvae_per_female_worm=larvae_per_female_worm,
            mating_probability=0.05,
        )
        self.k = k
        self.exposure_heterogeneity = np.random.gamma(
            shape=k, scale=1 / k, size=num_individuals
        )
        self.ages = np.full(num_individuals, 0)
        self.sink_name_order = list(sink_interaction_values.keys())
        self.sink_interaction = np.array(
            [sink_interaction_values[key] for key in self.sink_name_order]
        ).T

    def age(self, timestep: int):
        self.ages += timestep
        return self.worm_pop.age(timestep)

    def check_emergences(self, interaction_occured: list[bool]) -> float:
        return self.worm_pop.check_emergences(interaction_occured)

    def stats(self):
        num_infected_with_larvae = np.mean(np.sum(self.worm_pop.larvae, axis=1) > 0)
        num_infected_with_worm = np.mean(
            np.sum(self.worm_pop.getTotalWorms(), axis=1) > 0
        )

        print(
            f"Larval Infection prevalence: {num_infected_with_larvae}. Worm Infection prevalence: {num_infected_with_worm}"
        )
