library(data.table)
library(doParallel)
library(sunny)
library(clobe)
library(psp)
## set working directory to the location of this script
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("convert.R")
source("imac.R")
source("psp.R")
# load("dome21.RData")

stimuli <- c("A", "B", "BC", "C")
path <- "./"
hypersphere <- 1
upper <- c(5, 1, 1, 5, 5)
lower <- c(1, 0.001, 0.001, 0.001, 0.001)
init <- c(3, 0.59, 0.35, 2, 3)
iteration <- 10000
iteration <- 100
population <- 1000000
stimuli <- c("A", "B", "BC", "C")
hypersphere <- 1
pnames <- c("P", "alpha", "mu", "rho", "beta")
export_packages <- c(
    "catlearn", "data.table", "clobe", "sunny"
)
path <- "."

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

tracking <- data.table(ppt = unique(combined_data$ppt), explored = FALSE, started = "", completed = "")

rm(ply207, ply216, ply222, combined_data, repeat_participants)
gc()

data <- combined_data
i <- unique(data$ppt)[100]

jump <- apply(rbind(lower, upper), 2, function(x) {
    dis <- runif(1000, x[1], x[2])
    return(dis)
})

if (!("data.table" %in% is(data))) data <- data.table(data)
participant <- data[ppt == i]
tracking[ppt == i, started := as.POSIXct(Sys.time())]
frequencies <- data.table()


participant$phase <- as.integer(participant$phase == "test")
participant[, phase := phase * 2]
mode <- "supervised"
maximum_trials <- 20
if (unique(participant$dataset) == "ply207") {
    maximum_trials <- 12
}
if (unique(participant$dataset) == "ply216") {
    participant[phase == 0, abresp := "none"] ## make sure that no erroneous recordings are present
    participant[phase == 0 & abstim == "AB", abstim := "ABX"]
    participant[phase == 0 & abstim == "AC", abstim := "ACY"]
    mode <- "unsupervised"
} else if (unique(participant$dataset) == "ply222") {
    participant[phase == 0, abresp := "none"] ## make sure that no erroneous recordings are present
    participant[phase == 0 & abstim == "ABD", abstim := "ABX"]
    participant[phase == 0 & abstim == "ACE", abstim := "ACY"]
    mode <- "unsupervised"
}

trial_order <- participant[ppt == i]$abstim
phase_index <- participant[ppt == i]$phase
phase_index[phase_index == "training"] <- 0
phase_index[phase_index == "test"] <- 2
phase_index <- as.numeric(phase_index)
if (mode == "supervised") {
    phase_index[phase_index == 0] <- 4
}
trials <- participant[ppt == i]$trial
## convert the observed trial sequence into the model training format
train <- dome21train(
    schedule = trial_order,
    block = phase_index,
    trials = trials,
    ppt = i,
    design = "supervised"
)

train$ctrl[train$ctrl == 0] <- 4

## bind the current participant's training set into the PSP callbacks
.evaluate <- function(x) {
    out <- evaluate(params = x, stimuli = stimuli, tr = train)
    return(out)
}


## discretize continuous predictions into ordinal response patterns
.discretize <- function(x) {
    ## if any probabilities are NA, exchange all predictions for the current parameter set with NA to prevent errors in the pspGlobal exploration
    if (any(is.na(x))) {
        x <- rep(NA, length(x))
    }
    out <- discretize(probabilities = x, thresholds = c(0.125, -0.125))
    return(out)
}


explored_file <- paste(path, "/OSCAR_", length(stimuli), ".", i, sep = "")

## explore the participant-specific parameter space and save the raw CSV output
out <- pspGlobal(
    model = .evaluate, discretize = .discretize,
    control = list(
        radius = hypersphere,
        lower = lower, upper = upper,
        init = jump,
        dimensionality = length(stimuli) + 1,
        responses = length(stimuli) + 1,
        parameter_names = pnames,
        stimuli_names = stimuli,
        population = population,
        iterations = iteration
    ),
    quiet = FALSE,
    save = TRUE, path = explored_file, extension = ".csv"
)
partitions <- out$ordinal_patterns
partitions <- list(partitions)
names(partitions) <- i
## preserve the compact ordinal-pattern summary for later analyses
name <- paste("OSCAR_", length(stimuli), ".", i, sep = "")
output_file <- paste(path, "/OSCAR_", length(stimuli), ".", i, ".RData", sep = "")
assign(name, partitions)
save(list = name, file = output_file)
## save the frequency counts of each ordinal pattern for later analyses
frequencies <- rbind(
    frequencies,
    data.table(ppt = i, dataset = unique(participant$dataset), frequency = data.table(out$ordinal_counts))
)
fwrite(frequencies, paste(path, "/frequencies.csv", sep = ""))
## update the tracking data table to indicate that this participant's exploration is complete and save the updated tracking information
tracking[ppt == i, completed := Sys.time()]
tracking[ppt == i, explored := TRUE]
fwrite(tracking, paste(path, "/tracking.csv", sep = ""))

rm(partitions) ## clean up workspace
gc()
message <- paste("Completed participant ", i, " of ", length(unique(data$ppt)), sep = "")
print(message)
