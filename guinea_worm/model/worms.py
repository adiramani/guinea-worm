import numpy as np

class Worms:
    worm_death_rate: int
    death_prob_by_age: float
    sex_ratio: float = 0.5
    worm_maturity_age_days: int
    mating_probability: float
    emergences: list[int]
    mature_male_has_existed: list[list[int]]

    # Rows = Individuals, Columns = Worm/Larvae Age
    male_worms: list[list[int]]
    female_worms: list[list[int]]

    def __init__(
        self,
        worm_death_rate: float,
        max_worm_age: int,
        individuals: int,
        mating_probability: float,
        worm_maturity_age_days: int
    ):
        self.worm_death_rate = worm_death_rate
        self.male_worms = np.full((individuals, max_worm_age), 0)
        self.female_worms = np.full((individuals, max_worm_age), 0)
        self.mature_male_has_existed = np.full((individuals, max_worm_age), False)
        self.worm_maturity_age_days = worm_maturity_age_days
        prob_death_array = 1 - np.exp(-(self.worm_death_rate) * np.arange(0, max_worm_age))
        prob_death_array[:worm_maturity_age_days] = 0
        prob_death_array[-1] = 1
        self.death_prob_by_age = np.tile(prob_death_array, (individuals, 1))
        self.mating_probability = mating_probability
        self.emergences = np.zeros(individuals)

    def get_total_worms(self):
        return np.sum(self.male_worms, axis=1) + np.sum(self.female_worms, axis=1)
    
    def get_female_worm_burden(self):
        return np.sum(self.female_worms, axis=1)
    
    def get_mating_probability(self):
        if np.sum(self.female_worms) == 0:
            return 0
        return (
            np.sum(self.mature_male_has_existed) /
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

    def age(self, timestep: int) -> int:
        self.mature_male_has_existed = np.logical_or(
            self.mature_male_has_existed,
            np.tile((np.sum(self.male_worms, axis=1) > 0)[:, np.newaxis], (1, self.mature_male_has_existed.shape[1]))
        )
        # mature_male_existed = np.sum(self.male_worms, axis=1) > 0
        # mature_male_existed[:self.worm_maturity_age_days] = False
        female_worm_deaths = np.where(np.random.rand(*self.male_worms.shape) < self.death_prob_by_age, True, False)
        male_worm_deaths = np.where(np.random.rand(*self.female_worms.shape) < self.death_prob_by_age, True, False)
        
        dead_males = self.male_worms[male_worm_deaths]

        self.emergences = np.sum(np.where(
            np.logical_and(
                female_worm_deaths,
                self.mature_male_has_existed
            ),
            self.female_worms,
            0
        ), axis=1)

        # self.emergences = np.where(
        #     np.logical_and(
        #         dead_females,
        #         mature_male_existed
        #     ),
        #     dead_females,
        #     0
        # )

        self.male_worms[male_worm_deaths] = 0
        self.female_worms[female_worm_deaths] = 0

        self.male_worms = np.roll(self.male_worms, shift=timestep, axis=1)
        self.female_worms = np.roll(self.female_worms, shift=timestep, axis=1)
        self.mature_male_has_existed = np.roll(self.mature_male_has_existed, shift=timestep, axis=1)

        self.male_worms[:, :timestep] = 0
        self.female_worms[:, :timestep] = 0
        self.mature_male_has_existed[:, :timestep] = 0

        
    def worms_emerging(self, interaction_occured: list[bool]) -> float:
        emergences_occuring = self.emergences > 0
        number_of_female_worms_emerging = 0
        if (emergences_occuring[interaction_occured]).any():
            number_of_female_worms_emerging = np.sum(self.emergences[interaction_occured])

            self.emergences = np.zeros(len(self.emergences))
        return number_of_female_worms_emerging
