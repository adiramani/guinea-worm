import numpy as np


class Worms:
    _max_male_age: int
    _max_female_age: int
    sex_ratio: float = 0.5
    max_larval_age: int
    larvae_per_female_worm: float
    mating_probability: float
    larval_release_rate: float
    emergences: list[int]

    # Rows = Individuals, Columns = Worm/Larvae Age
    male_worms: list[list[int]]
    female_worms: list[list[int]]
    larvae: list[list[int]]

    time_in_presence_of_male: list[list[int]]

    def __init__(
        self,
        max_worm_age: int,
        max_larval_age: int,
        individuals: int,
        larval_release_rate: float,
        larvae_per_female_worm: float,
        mating_probability: float,
    ):
        self._max_male_age = self._max_female_age = max_worm_age
        self.male_worms = np.full((individuals, max_worm_age), 0)
        self.female_worms = np.full((individuals, max_worm_age), 0)
        self.time_in_presence_of_male = np.full((individuals, max_worm_age), 0)

        self.max_larval_age = max_larval_age
        self.larvae = np.full((individuals, max_larval_age), 0)
        self.larval_release_rate = larval_release_rate

        self.larvae_per_female_worm = larvae_per_female_worm
        self.mating_probability = mating_probability

    def _age_helper(self, worm_arr, timestep):
        shifted_worms = np.zeros(worm_arr.shape, dtype=int)
        shifted_worms[:, timestep:] = worm_arr[:, :-timestep]
        return shifted_worms

    def getTotalWorms(self):
        return self.male_worms + self.female_worms

    def age(self, timestep: int) -> int:

        female_worm_deaths = np.sum(self.female_worms[:, -timestep:], axis=1)
        time_with_male_present = np.sum(
            self.time_in_presence_of_male[:, -timestep:], axis=1
        )
        mating_probability_at_death = np.minimum(
            time_with_male_present * self.mating_probability, 1.0
        )

        self.emergences = np.where(
            (female_worm_deaths > 0)
            & (
                np.random.rand(len(mating_probability_at_death))
                < mating_probability_at_death
            ),
            female_worm_deaths,
            0,
        )

        self.male_worms = self._age_helper(self.male_worms, timestep)
        self.female_worms = self._age_helper(self.female_worms, timestep)
        self.time_in_presence_of_male = self._age_helper(
            self.time_in_presence_of_male, timestep
        )

        larvae_to_worms = np.sum(self.larvae[:, -timestep:], axis=1)
        self.larvae = self._age_helper(self.larvae, timestep)

        new_male_worms = np.random.binomial(larvae_to_worms, self.sex_ratio)
        new_female_worms = larvae_to_worms - new_male_worms
        self.male_worms[:, 0] = new_male_worms
        self.female_worms[:, 0] = new_female_worms

        male_worm_exists_by_individual = np.sum(self.male_worms, axis=1) > 0
        male_worm_exists = np.tile(
            male_worm_exists_by_individual, (self.male_worms.shape[1], 1)
        ).T
        female_worm_exists = self.female_worms > 0
        female_with_male_mask = np.where(male_worm_exists & female_worm_exists)[0]
        if female_with_male_mask.any():
            self.time_in_presence_of_male[female_with_male_mask] += 1

    def check_emergences(self, interaction_occured: list[bool]) -> float:
        if (self.emergences[interaction_occured] > 0).any():
            return_val = (
                self.emergences[interaction_occured] * self.larvae_per_female_worm
            )
            self.emergences = np.zeros(len(self.emergences))
            return np.sum(return_val) * self.larvae_per_female_worm
        return 0
