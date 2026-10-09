library(data.table)
library(doParallel)
library(sunny)
library(clobe) # https://github.com/lenarddome/clobe
source("convert.R")
source("imac.R")
source("psp.R")


stimuli <- c("A", "B", "BC", "C")
path <- "results"
hypersphere <- 1
upper <- c(5, 1, 1, 5, 5)
lower <- c(1, 0.001, 0.001, 0.001, 0.001)
init <- c(3, 0.59, 0.35, 2, 3)
iteration <- 10000

ply207 <- fread("../data/ply207-canonical-ibre.csv")
ply216 <- fread("../data/ply216-observational-ibre.csv")
ply222 <- fread("../data/ply222-non-causal-ibre.csv")

ply207[, "dataset" := "ply207"]
ply216[, "dataset" := "ply216"]
ply222[, "dataset" := "ply222"]

## ply207 encodes abresp as 0/1/2, convert to the shared common/rare/none format
ply207$abresp <- factor(as.character(ply207$abresp), levels = c("0", "1", "2"), labels = c("common", "rare", "none"))

combined_data <- rbind(ply216, ply222, use.names = TRUE, fill = TRUE)
combined_data <- rbind(combined_data[, c("abresp", "abstim", "phase", "trial", "ppt", "dataset")], ply207[, c("abresp", "abstim", "phase", "trial", "ppt", "dataset")])

repeat_participants <- combined_data[, length(unique(dataset)), by = .(ppt)][V1 > 1]$ppt
## participants that appear in more than one dataset should have different ids
for (ppt in repeat_participants) {
    indices <- which(combined_data$ppt == ppt)
    combined_data$ppt[indices] <- paste0(combined_data$ppt[indices], "_", combined_data$dataset[indices])
}

data <- combined_data
rm(ply207, ply216, ply222, combined_data, repeat_participants)
gc()

pspNEW(
    stimuli = stimuli,
    path = path,
    hypersphere = hypersphere,
    upper = upper,
    lower = lower,
    init = init,
    iteration = iteration,
    data = data
)
