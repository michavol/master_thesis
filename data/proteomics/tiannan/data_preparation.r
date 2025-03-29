load("data/prepData.RData")
n_protein <- 5519

dat <- data[data$pertLabel != 'no no' & data$pertLabel != 'no', ]

dat[which(is.na(dat[, 'IC50'])), 'Sample_ID']
names(data)[n_protein + 0:2]
names(data)[1]

library(dplyr)
library(tidyr)

#aggregate data to median/mean per protein per cell line per drug per dose per time point
dat <- data[, -which(names(data) %in% c('type', 'Sample_ID', 'BioRep', 
                                        'machine', 'NY'))]

dat %>% group_by(protein_plate, pertLabel, Anchor_dose, Library_dose, pert_time) %>% 
    summarise_all(funs(median)) -> agg_data
dim(agg_data)

#combine different pert_time into one row per cell line per drug per dose
prot_names <- names(data)[1:n_protein]
n_pert_single <- length(unique(data$pertLabel[data$type == 'singleDrug']))
n_pert_single
pert_names <- names(data)[(1+n_protein):(n_protein + n_pert_single)]

baseline_prot <- agg_data[agg_data$pertLabel == 'no', ]
agg_data <- agg_data[agg_data$pertLabel != 'no no', ]
agg_data <- agg_data[agg_data$pertLabel != 'no', ]

dim(baseline_prot)
baseline_prot

#use only cell lines with baseline data
unique(agg_data$protein_plate)[!(unique(agg_data$protein_plate) %in% 
                                   baseline_prot$protein_plate)]
agg_data <- agg_data[agg_data$protein_plate %in% baseline_prot$protein_plate, ]

#add baseline to each perturbation
pert_info_names <- c('protein_plate', 'pertLabel', 'Anchor_dose', 'Library_dose', 
                     'IC50', pert_names)
baseline <- agg_data[, pert_info_names]
dim(baseline)
dim(unique(baseline))
baseline <- unique(baseline)
baseline$pert_time <- 0

for (i in 1:nrow(baseline_prot)){
  baseline[baseline$protein_plate == 
             as.character(baseline_prot[i, 'protein_plate']), prot_names] <- 
    baseline_prot[i, prot_names]
}

agg_data <- rbind(agg_data, baseline)
agg_data %>% pivot_wider(names_from = pert_time, values_from = all_of(prot_names)) -> agg_data
sum(is.na(agg_data[, 3]))
na.omit(agg_data) -> agg_data #remove rows with not all three time points

agg_data$type <- 'singleDrug'
agg_data$type[agg_data$Anchor_dose != 0] <- 'drugCombination'

save(agg_data, file = 'data/aggData.RData')

dat <- data[, -which(names(data) %in% c('type', 'Sample_ID', 'NY'))]
dat <- dat[dat$pertLabel != 'no no' & dat$pertLabel != 'no', ]

dat <- dat[dat$protein_plate %in% baseline_prot$protein_plate, ]
baseline <- dat

for (i in 1:nrow(baseline_prot)){
    baseline[baseline$protein_plate == as.character(baseline_prot[i, 'protein_plate']), prot_names] <- baseline_prot[i, prot_names]
}

names(baseline)[1:n_protein] <- paste0(prot_names, '_baseline')


comb_data <- cbind(baseline[, 1:n_protein], dat)
comb_data <- na.omit(comb_data)

comb_data$type <- 'singleDrug'
comb_data$type[comb_data$Anchor_dose != 0] <- 'drugCombination'

save(comb_data, file = 'data/combData.RData')

prot_names
prot_names_baseline <- paste0(prot_names, '_baseline')
prot_names_0 <- paste0(prot_names, '_0')
prot_names_6 <- paste0(prot_names, '_6')
prot_names_24 <- paste0(prot_names, '_24')
prot_names_48 <- paste0(prot_names, '_48')

save(prot_names, prot_names_baseline, prot_names_0, prot_names_6, prot_names_24, prot_names_48, pert_names, file = 'data/protNames.RData')


## aggData with differences between timepoints
agg_data_diff <- agg_data
agg_data_diff[, prot_names_48] <- agg_data_diff[, prot_names_48] - agg_data_diff[, prot_names_24]
agg_data_diff[, prot_names_24] <- agg_data_diff[, prot_names_24] - agg_data_diff[, prot_names_6]
agg_data_diff[, prot_names_6] <- agg_data_diff[, prot_names_6] - agg_data_diff[, prot_names_0]

save(agg_data_diff, file = 'data/aggData_diff.RData')
