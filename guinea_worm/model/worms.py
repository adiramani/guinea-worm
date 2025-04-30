import numpy as np
import scipy.stats as stats

class Worms:
    worm_death_rate: int
    death_prob_by_age: float
    sex_ratio: float = 0.5
    worm_maturity_age_days: int
    worm_death_gamma_shape: float
    mating_probability: float
    emergences: list[int]
    total_female_worm_age_at_emergence: int
    max_worm_age: int
    mature_male_has_existed: list[list[int]]

    # Rows = Individuals, Columns = Worm/Larvae Age
    male_worms: list[list[int]]
    female_worms: list[list[int]]

    def __init__(
        self,
        timestep: int,
        worm_death_rate: float,
        max_worm_age: int,
        individuals: int,
        mating_probability: float,
        worm_maturity_age_days: int,
        worm_death_gamma_shape: float
    ):
        self.worm_death_rate = worm_death_rate
        self.max_worm_age = max_worm_age
        self.male_worms = np.full((individuals, max_worm_age //  timestep), 0)
        self.female_worms = np.full((individuals, max_worm_age //  timestep), 0)
        self.mature_male_has_existed = np.full((individuals, max_worm_age // timestep), False)
        self.worm_maturity_age_days = worm_maturity_age_days // timestep
        self.worm_death_gamma_shape = worm_death_gamma_shape
        valid_worm_ages = np.arange(start=0, stop=max_worm_age, step=timestep)
        pdf = stats.gamma.pdf(valid_worm_ages, a=worm_death_gamma_shape, scale=(1/(worm_death_gamma_shape*self.worm_death_rate)))
        cdf = stats.gamma.cdf(valid_worm_ages, a=worm_death_gamma_shape, scale=(1/(worm_death_gamma_shape*self.worm_death_rate)))
        hazards = pdf / (1 - cdf)
        prob_death_array = 1 - np.exp(-hazards * valid_worm_ages)
        prob_death_array = 1 - np.exp(-hazards * timestep)

        self.death_prob_by_age = np.tile(prob_death_array, (individuals, 1))
        self.death_prob_by_age[:, -1] = 1

        self.mating_probability = mating_probability
        self.emergences = np.zeros(individuals)
        self.total_female_worm_age_at_emergence = 0

    def get_total_worms(self):
        return np.sum(self.male_worms, axis=1) + np.sum(self.female_worms, axis=1)
    
    def get_female_worm_burden(self):
        return np.sum(self.female_worms, axis=1)
    
    def get_mating_probability(self):
        if np.sum(self.female_worms) == 0:
            return 0
        return (
            np.sum(
                np.where(
                    self.mature_male_has_existed,
                    self.female_worms,
                    0
                )
            ) /
            np.sum(self.female_worms)
        )
    
    def new_worms_injested(self, new_worms: list[int]):
        new_male_worms = np.random.binomial(new_worms, self.sex_ratio)
        new_female_worms = new_worms - new_male_worms
        self.male_worms[:, 0] += new_male_worms
        self.female_worms[:, 0] += new_female_worms
    
    def process_host_death(self, individuals: list[bool]):
        self.male_worms[individuals, :] = 0
        self.female_worms[individuals, :] = 0

    def age(self, timestep: int, current_time: int, burnin_time: int) -> int:
        self.mature_male_has_existed = np.logical_or(
            self.mature_male_has_existed,
            np.tile((np.sum(self.male_worms, axis=1) > 0)[:, np.newaxis], (1, self.mature_male_has_existed.shape[1]))
        )
        
        female_worm_deaths = np.random.binomial(self.female_worms, self.death_prob_by_age)

        male_worm_deaths = np.random.binomial(self.male_worms, self.death_prob_by_age)

        emergence_by_age = None
        if current_time < (burnin_time):
            emergence_by_age = female_worm_deaths
        else:
            emergence_by_age = np.where(
                self.mature_male_has_existed,
                female_worm_deaths,
                0
            )

        self.emergences += np.sum(emergence_by_age, axis=1)

        self.total_female_worm_age_at_emergence = np.sum(
            np.sum(emergence_by_age, axis=0) * np.arange(start=0, stop=self.max_worm_age, step=timestep)
        )

        self.male_worms = self.male_worms - male_worm_deaths
        self.female_worms = self.female_worms - female_worm_deaths

        self.male_worms = np.roll(self.male_worms, shift=1, axis=1)
        self.female_worms = np.roll(self.female_worms, shift=1, axis=1)
        self.mature_male_has_existed = np.roll(self.mature_male_has_existed, shift=1, axis=1)

        self.male_worms[:, 0] = 0
        self.female_worms[:, 0] = 0
        self.mature_male_has_existed[:, 0] = False

        
    def worms_emerging(self, interaction_occured: list[bool], teathering_efficacy: float) -> tuple[float, float, float]:
        number_of_female_worms_emerging = 0
        number_of_dogs_with_emerging_worms = 0
        is_not_teathered = np.random.rand() >= teathering_efficacy
        if (self.emergences[interaction_occured]).any():
            number_of_female_worms_emerging = np.sum(
                self.emergences[
                    np.logical_and(
                        interaction_occured,
                        is_not_teathered
                    )
                ]
            )
            number_of_dogs_with_emerging_worms = np.sum(
                self.emergences[
                    np.logical_and(
                        interaction_occured,
                        is_not_teathered
                    )
                ] > 0
            )
        total_age = self.total_female_worm_age_at_emergence

        self.emergences = np.zeros(len(self.emergences))
        self.total_female_worm_age_at_emergence = 0
        return number_of_female_worms_emerging, number_of_dogs_with_emerging_worms, total_age
