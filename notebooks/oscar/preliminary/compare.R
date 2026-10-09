# The participant-level OSCAR routine (psp.R) for the first N participants of ply207 with
# reduced settings, timed, with the patterns found saved for comparison with the torch port.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; dir <- args[2]; repo <- args[3]; out <- args[4]
n_ppt <- as.integer(args[5]); starts <- as.integer(args[6]); iterations <- as.integer(args[7])
.libPaths(c(lib, .libPaths()))
suppressMessages({library(Rcpp); library(data.table); library(clobe); library(psp)})
dir.create(out, showWarnings = FALSE, recursive = TRUE)
setwd(dir)
Rcpp::sourceCpp("oscar_all.cpp", verbose = FALSE)
source(file.path(repo, "tmp-models", "convert.R"))
source(file.path(repo, "tmp-models", "imac.R"))
data <- fread(file.path(repo, "tmp-models", "ply207-canonical-ibre.csv"))
data$abresp <- factor(as.character(data$abresp), levels = c("0", "1", "2"), labels = c("common", "rare", "none"))
ppts <- unique(data$ppt)[seq_len(n_ppt)]
stimuli <- c("A", "B", "BC", "C")
lower <- c(1, 0.001, 0.001, 0.001, 0.001); upper <- c(5, 1, 1, 5, 5)
set.seed(1)
jump <- apply(rbind(lower, upper), 2, function(x) runif(starts, x[1], x[2]))   # as psp.R: one matrix for all
summary <- NULL; patterns <- NULL
for (i in ppts) {
  participant <- data[ppt == i]
  participant$phase <- as.integer(participant$phase == "test")
  participant[, phase := phase * 2]
  phase_index <- as.numeric(participant$phase); phase_index[phase_index == 0] <- 4
  train <- ConvertToStateList(schedule = participant$abstim, block = phase_index, trials = participant$trial, ppt = i, design = "supervised")
  .evaluate <- function(x) evaluate(params = x, stimuli = stimuli, tr = train)
  .discretize <- function(x) { if (any(is.na(x))) x <- rep(NA, length(x)); discretize(probabilities = x, thresholds = c(0.125, -0.125)) }
  set.seed(2026)
  seconds <- system.time(
    res <- pspGlobal(model = .evaluate, discretize = .discretize,
      control = list(radius = 1, lower = lower, upper = upper, init = jump, dimensionality = 5, responses = 5,
                     parameter_names = c("P", "alpha", "mu", "rho", "beta"), stimuli_names = stimuli,
                     population = 1000000, iterations = iterations),
      quiet = TRUE)
  )[["elapsed"]]
  n <- dim(res$ordinal_patterns)[3]
  summary <- rbind(summary, data.frame(ppt = i, seconds = seconds, iterations = res$iterations, patterns = n,
                                       evaluations = sum(res$ordinal_counts)))
  cat(i, ":", seconds, "s,", n, "patterns,", sum(res$ordinal_counts), "evaluations\n")
  patterns <- rbind(patterns, data.frame(ppt = i, id = seq_len(n), count = as.vector(res$ordinal_counts),
                                         key = apply(res$ordinal_patterns, 3, function(m) paste(m[upper.tri(m)], collapse = ","))))
}
write.csv(summary, file.path(out, "r_summary.csv"), row.names = FALSE)
write.csv(patterns, file.path(out, "r_patterns.csv"), row.names = FALSE)
cat("done\n")
