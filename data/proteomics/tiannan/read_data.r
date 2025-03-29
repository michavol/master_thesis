# Set the working directory to the R_data folder
setwd("/home/mike/Masters_DS/master_thesis/data/proteomics/R_data")

# Load the aggData.RData file
load("aggData.RData")

# Display the content of the loaded object
agg_data

# Save as a CSV file
write.csv(agg_data, "aggData.csv", row.names = FALSE)

# Print structure of agg_data
str(agg_data)