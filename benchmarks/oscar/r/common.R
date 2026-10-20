# Shared setup of the R side: the R psp package, the compiled C++ OSCAR (RcppArmadillo) and the
# helpers of the participant-level routine (convert.R, imac.R), as psp.R / run.r use them.
# Sourced with `lib` (an R library holding psp) and `models` (the tmp-models directory) defined.
#
# OSCAR is the model of Dome and Wills (2026), https://doi.org/10.48550/arXiv.2608.06149.
.libPaths(c(lib, .libPaths()))
suppressMessages({library(Rcpp); library(RcppArmadillo); library(data.table); library(clobe); library(psp)})

# architecture.cpp and OSCAR.cpp in one translation unit, so that sourceCpp() can build them
build <- file.path(tempdir(), "oscar-build")
dir.create(build, showWarnings = FALSE)
file.copy(file.path(models, "architecture.h"), build, overwrite = TRUE)
writeLines(c(readLines(file.path(models, "architecture.cpp")), readLines(file.path(models, "OSCAR.cpp"))),
           file.path(build, "oscar_all.cpp"))
Rcpp::sourceCpp(file.path(build, "oscar_all.cpp"), verbose = FALSE, cacheDir = file.path(build, "cache"))
source(file.path(models, "convert.R"))
source(file.path(models, "imac.R"))

ibre <- fread(file.path(models, "ply207-canonical-ibre.csv"))
ibre$abresp <- factor(as.character(ibre$abresp), levels = c("0", "1", "2"), labels = c("common", "rare", "none"))
stimuli <- c("A", "B", "BC", "C")
lower <- c(1, 0.001, 0.001, 0.001, 0.001)
upper <- c(5, 1, 1, 5, 5)

# The trial matrix of one participant, as psp.R builds it (every participant here is supervised).
participant_training <- function(i) {
  participant <- ibre[ppt == i]
  participant$phase <- as.integer(participant$phase == "test")
  participant[, phase := phase * 2]
  phase_index <- as.numeric(participant$phase)
  phase_index[phase_index == 0] <- 4
  ConvertToStateList(schedule = participant$abstim, block = phase_index, trials = participant$trial,
                     ppt = i, design = "supervised")
}

discretize_na <- function(x) {
  if (any(is.na(x))) x <- rep(NA, length(x))
  discretize(probabilities = x, thresholds = c(0.125, -0.125))
}
