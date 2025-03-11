library(dplyr)
library(fastDummies)

na_imputation <- function(x){
  x[is.na(x)] <- min(x, na.rm = T) * 0.8
  return(x)
}

dat <- read.csv('../data/ProteinMatrix_sampleID_MapEC50_20240229.csv')
# why are the no IC50 values for some single drug experiments?
dat[dat$pert_id == dAB[472] & is.na(dat$EC50), 5580:ncol(dat)]

#### protein data ####
# select protein columns
p_idx <- 2:5586

print(names(dat)[1:p_idx[1]])
print(names(dat)[p_idx[length(p_idx)]:(p_idx[length(p_idx)]+2)])

data_protein <- dat[, p_idx]

# remove columns with only one value
idx <- which(apply(data_protein, 2, function(x) length(unique(x))) == 1)
data_protein <- data_protein[, -idx]
data_protein <- data_protein[, grepl("HUMAN", names(data_protein))]

na_count <- apply(data_protein, 2, function(x) sum(is.na(x)))
na_count <- na_count / nrow(data_protein)
hist(na_count, breaks = 100)
max(na_count)

mean(is.na(data_protein))

save(na_count, file = '../data/na_count.RData')

hist(log(apply(data_protein, 2, function(x) quantile(x, 0.2, na.rm = T))), breaks = 100)
hist(log(apply(data_protein, 2, function(x) quantile(x, 0.5, na.rm = T))), breaks = 100)
hist(log(apply(data_protein, 2, function(x) quantile(x, 0.7, na.rm = T))), breaks = 100)
hist(log(apply(data_protein, 2, function(x) quantile(x, 0.9, na.rm = T))), breaks = 100)

# impute missing values
data_protein <- apply(data_protein, 2, na_imputation)
#data_protein[is.na(data_protein)] <- 0

print('missing values')
print(sum(is.na(data_protein)))

# log transform
data_protein <- log(data_protein)

#### drug data ####
drugs <- dat$pert_id
data_drugs <- dummy_cols(data.frame(drug = drugs))[, -1]

# no drug as reference, remove column
#data_drugs <- data_drugs[, -which(colnames(data_drugs) == "drug_no")]

#drug combinations
combination_idx <- which(data_drugs[, "drug_"] == 1)
#drug_combinations <- dat[combination_idx, 'drugIdAB']
#drug_combinations

#drugs_in_combinations <- unique(unlist(lapply(drug_combinations, function(combination) {
#  strsplit(combination, " ")[[1]]
#})))

#drugs_in_combinations <- paste('drug', drugs_in_combinations, sep = "_")
#if(any(!drugs_in_combinations %in% colnames(data_drugs))){
#  missing_drugs <- drugs_in_combinations[!drugs_in_combinations %in% colnames(data_drugs)]
#  data_drugs <- cbind(matrix(0, nrow = nrow(data_drugs), ncol = length(missing_drugs)), data_drugs)
#  colnames(data_drugs)[1:length(missing_drugs)] <- missing_drugs
#}

#### drug conbinations consentrations
#data_cons <- dat[combination_idx, c('Anchor_dose', 'Library_dose')]
#data_cons <- data_cons / rowSums(data_cons)

#data_cons

#data_combination <- sapply(drug_combinations, function(combination) {
#    drugAB <- strsplit(combination, " ")[[1]]
#    drugAB <- paste('drug', drugAB, sep = "_")
#    as.numeric(colnames(data_drugs) %in% drugAB)
#})

#data_combination <- t(data_combination)
#colnames(data_combination) <- colnames(data_drugs)

#data_drugs[combination_idx, ] <- data_combination

data_drugs <- data_drugs * 10

for (i in combination_idx) {
  drugAB <- strsplit(dat[i, 'drugIdAB'], " ")[[1]]
  drugAB <- paste('drug', drugAB, sep = "_")
  
  data_drugs[i, drugAB[1]] <- dat[i, 'Anchor_dose']
  data_drugs[i, drugAB[2]] <- dat[i, 'Library_dose']
}



# remove unused drugs
names(data_drugs)

data_drugs <- data_drugs[, -which(colnames(data_drugs) == "drug_")] # dummy for combination
data_drugs <- data_drugs[, -which(colnames(data_drugs) == "drug_no")] # no drug as reference

#TODO: what about nan and medium?
#TODO: what about anchor?
#TODO: what about combination concentration?

#### additonal data ####
data_additional <- dat[, c('pert_time', 'protein_plate', 'machine', 'BioRep', 'Sample_ID', 'Anchor_dose', 'Library_dose')]
data_additional$Anchor_dose[is.na(data_additional$Anchor_dose)] <- 0
data_additional$Library_dose[is.na(data_additional$Library_dose)] <- 0

#### response data ####
data_response <- dat$EC50
data_response[combination_idx] <- dat$Combo.IC50[combination_idx]

#### combine data ####
type <- rep('singleDrug', nrow(data_protein))
type[combination_idx] <- 'drugCombination'
type[which(drugs == 'no')] <- 'noDrug'

pertLabel <- dat$pert_id
pertLabel[combination_idx] <- dat$drugIdAB[combination_idx]
#pertLabel[pertLabel == 'no no'] <- 'no'

# no drug
data_response[which(pertLabel == 'no')] <- Inf

data <- cbind(data_protein, data_drugs, data_additional, type, pertLabel, IC50 = data_response, NY = dat$NY)

#log transform of single drug
data[data$type == 'singleDrug', 'IC50'] <- log2(data[data$type == 'singleDrug', 'IC50'])

plot(as.factor(data$type), data$IC50, ylab = 'log2(IC50)')

#save data
save(data, file = '../data/prepData.RData')
