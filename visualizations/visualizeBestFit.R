library(ggplot2)
library(dplyr)

observed_data <- read.csv("/Users/adi/Documents/RVC/Guinea Worm/carter center data/clean/case_data_single_case.csv") %>% 
filter(
  snu1_detect == "Cb"
) %>%
mutate(
    year = lubridate::year(as.Date(date_emerge)),
    month = lubridate::month(as.Date(date_emerge))
) %>%
filter(
    year >= 2014 & year <= 2017
) %>%
group_by(year, month) %>%
summarise(cases = sum(worm_emerge), .groups="drop") %>%
mutate(
    year = (year - min(year)) + ((month - 1) / 12)
)

desc = ""
its = 10

best_fit_info <- read.csv(paste0("../output_data", desc, "_", its, "_its/fit_results/fit_results.csv")) %>%
    group_by(r0, r0_sink_to_worm, nh_nc, exposure_heterogeneity) %>%
    summarise(mse = sum(mse), .groups="drop")
best_fit_overall <- best_fit_info[which(best_fit_info$mse == min(best_fit_info$mse)),]
best_fit_overall


filter_data <- function(data, prev=FALSE) {
    if(prev) {
        return(
        data %>% mutate(
                year = year - 1/12
            ) %>%
            filter(measure == "emergences_per_host_copepod" & year >= 15 & year < 19) %>%
            mutate(
                year = year - min(year)
            )
        )
    }
    return(
        data %>% mutate(
            year = year - 1/12
        ) %>%
        filter(measure == "emergence_copepod" & year >= 15 & year < 19) %>%
        mutate(
            year = year - min(year)
        )
    )
}

read_all_data <- function(iterations, description, best_fit_data, use_prev=FALSE) {
    all_data <- c()
    for (i in seq(0, iterations-1, 1)) {
        file_name = paste0(
            "../output_data", description, "_", iterations, "_its/",
            "model_output_initinf_1_nhnc_", best_fit_data[1, "nh_nc"], "_r0_",
            format(best_fit_data[1, "r0"], nsmall = 1), "_r0sw_", round(best_fit_data[1, "r0_sink_to_worm"], 2), "_runnum_",
            i, "_exposhet_", best_fit_data[1, "exposure_heterogeneity"], ".csv"
        )
        if (length(all_data) == 0) {
            all_data <- read.csv(file_name) %>% filter_data(prev=use_prev)
        } else {
            all_data <- rbind(all_data, read.csv(file_name) %>% filter_data(prev=use_prev))
        }
    }
    return(all_data)
}

best_fit_data <- read_all_data(its, desc, best_fit_overall %>% as.data.frame())

best_fit_data_summary <- best_fit_data %>% 
    group_by(year, measure, r0, r0_sink_to_worm, nh_nc, exposure_heterogeneity) %>%
    summarise(value = mean(value), .groups="drop")



ggplot() +
    geom_line(
        data = best_fit_data_summary,
        aes(x=year, y=value)
    ) +
    geom_line(
        data = observed_data,
        aes(x=year, y=cases),
        color="red"
    )
