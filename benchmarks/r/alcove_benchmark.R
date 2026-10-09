# The ALCOVE workload of benchmarks/alcove_benchmark.py, for the R package: ALCOVE
# on the full ranges of the learning rates, 20 random starting points, radius 0.15,
# population 50, a cap of 1,000 iterations, catlearn's trial order for seed 7624.
#
# Usage: Rscript benchmarks/r/alcove_benchmark.R <library path> <label> <seeds> <output directory> [iterations]
# <seeds> is comma-separated. Writes alcove.csv, alcove_discovery.csv and
# alcove_patterns.csv to the output directory, with the columns of the Python
# script, to be merged with `alcove_benchmark.py --import-r <output directory>`.

args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
label <- args[2]
seeds <- as.integer(strsplit(args[3], ",")[[1]])
out_dir <- args[4]
iterations <- if (length(args) >= 5) as.integer(args[5]) else 1000
suppressMessages(library(psp))
suppressMessages(library(catlearn))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## the model of the ALCOVE recipe, with the learning rates free up to 1
root <- dirname(dirname(dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))))))
source(file.path(root, "psp-reference", "vignettes", "articles", "alcove", "alcove-model.R"))
alcove_parameters$upper <- c(20, 6, 1, 1)
training <- alcove_training(learners = 5, seed = 7624)
model <- function(par) alcove_curves(par, training)
discretize <- function(curves) weak_order(curves, gap = 0.025)
types <- c("I", "II", "III", "IV", "V", "VI")
control <- list(
  lower = rep(0, 4), upper = rep(1, 4), radius = 0.15, population = 50, iterations = iterations,
  parameter_names = alcove_parameters$name,
  stimuli_names = paste0(rep(types, each = 16), "_", rep(1:16, 6)),
  dimensionality = 6, responses = 96
)

summary <- NULL
discovery <- NULL
found <- NULL
path <- file.path(tempdir(), "alcove_benchmark")
for (seed in seeds) {
  set.seed(seed)
  control$init <- matrix(runif(4 * 20), ncol = 4)
  seconds <- system.time(
    out <- pspGlobal(model, discretize, control, save = TRUE, path = path, quiet = TRUE)
  )[["elapsed"]]
  row <- data.frame(
    build = label, seed = seed, seconds = seconds, iterations = out$iterations,
    patterns = dim(out$ordinal_patterns)[3], evaluations = sum(out$ordinal_counts),
    filled = all(out$ordinal_counts >= 50)
  )
  cat(label, "seed", seed, ":", seconds, "s,", row$iterations, "iterations,", row$patterns,
      "patterns,", row$evaluations, "evaluations\n")
  summary <- rbind(summary, row)
  saved <- read.csv(paste0(path, "_parameters.csv"))
  seen <- integer(0)
  evaluations <- 0
  for (it in sort(unique(saved$iteration))) {
    rows <- saved$pattern[saved$iteration == it]
    seen <- union(seen, rows)
    evaluations <- evaluations + length(rows)
    discovery <- rbind(discovery, data.frame(build = label, seed = seed, iteration = it,
                                             evaluations = evaluations, patterns = length(seen)))
  }
  found <- rbind(found, data.frame(
    build = label, seed = seed,
    order = apply(out$ordinal_patterns, 3, describe_order),
    count = as.vector(out$ordinal_counts)
  ))
}
write.csv(summary, file.path(out_dir, "alcove.csv"), row.names = FALSE)
write.csv(discovery, file.path(out_dir, "alcove_discovery.csv"), row.names = FALSE)
write.csv(found, file.path(out_dir, "alcove_patterns.csv"), row.names = FALSE)
cat("done\n")
