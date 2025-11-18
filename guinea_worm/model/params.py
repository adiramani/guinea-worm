from dataclasses import dataclass, field

from guinea_worm.model.intervention import Intervention, InterventionEvent

@dataclass
class BaseParams():
    timestep: int = 5


@dataclass
class ModelParams(BaseParams):
    time: int = 0
    endtime: int = 20
    _days_in_year: int = 360

    host_populations: dict = field(default_factory=dict) # [str, HostPopulation]
    sink_populations: dict = field(default_factory=dict) # [str, SinkPopulation]
    interventions: dict[InterventionEvent, Intervention] = field(default_factory=dict)
    r0: float = 2.96
    verbose: bool = False
    emergence_events: dict[str, dict[str, int]] = field(default_factory=dict)
    burnin_time_years: int = 10
    a: float = 0.0089
    year_over_year_r0_factor: list[float] = field(
        default_factory=lambda: [
            1.0000000, 1.0916366, 0.9656724, 1.0484509, 1.0605562, 1.1621860, 1.1085350,
            0.9646439, 1.0912052, 1.0234629, 1.0729737
        ]
        
    )
    seasonality_start: int = 24
    beta_dist_z: list[float] = field(
        default_factory=lambda: [
            0.2905937, 0.3003394, 0.2746661, 0.2736649, 0.3142251, 0.2927766,
            0.2986853, 0.3010943, 0.2773236, 0.2949693, 0.3187633
        ]
    )
    beta_dist_q: list[float] = field(
        default_factory=lambda: [
            6.890029, 6.676211, 7.758946, 7.080747, 7.476747, 5.837766,
            6.541745, 8.465085, 7.161786, 7.060081, 6.971174
        ]
    )
    beta_dist_b: list[float] = field(
        default_factory=lambda: [
            0.1742691, 0.2000000, 0.2000000, 0.2000000, 0.2000000, 0.2000000,
            0.2000000, 0.2000000, 0.2000000, 0.1949314, 0.2000000
        ]
    )
    # beta_dist_mu: list[float] = field(
    #     default_factory=lambda: [
    #         0.3453301, 0.3306663, 0.3091962, 0.3059216, 0.3460586,
    #         0.3201639, 0.3239641, 0.3427088, 0.2981162, 0.3350126,
    #         0.3495668
    #     ]
    # )
    # beta_dist_kappa: list[float] = field(
    #     default_factory=lambda: [
    #         14.42108, 11.75474, 15.01853, 12.42899, 14.01285,
    #         10.42928, 15.64332, 21.96145, 13.23081, 16.27257,
    #         16.36524
    #     ]
    # )

@dataclass
class HostParams(BaseParams):
    num_individuals: int = 800
    population_name: str = "Dogs"
    mortality_rate: float = (2/5)/360
    initial_infected: int = 1
    worms_to_infect_with: int = 1
    worm_death_rate: float = (12/11)/360
    worm_death_gamma_shape: float = 37
    worm_mating_probability: float = 1
    k: float = 0.26
    sink_interaction_values: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    worm_maturity_age_days: int = 0
    max_worm_age: int = 720

@dataclass
class SinkParams(BaseParams):
    density: float = 0
    size: float = 0
    population_name: str = "copepod"
    infectivity_rate: float = 0
    larval_death_rate: int = 1/30