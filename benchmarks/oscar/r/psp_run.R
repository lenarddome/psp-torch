# The participant-level PSP routine (psp.R) for the first N participants with reduced settings,
# with the participants spread over `workers` forked processes (psp.R uses foreach over
# detectCores() - 2 cores). Writes one row per participant and the total wall-clock time.
#
#   Rscript psp_run.R <lib> <tmp-models dir> <out dir> <participants> <starts> <iterations> <workers>
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; models <- args[2]; out <- args[3]
n_ppt <- as.integer(args[4]); starts <- as.integer(args[5]); iterations <- as.integer(args[6])
workers <- as.integer(args[7])
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
library(parallel)
dir.create(out, showWarnings = FALSE, recursive = TRUE)

ppts <- unique(ibre$ppt)[seq_len(n_ppt)]
set.seed(1)
jump <- apply(rbind(lower, upper), 2, function(x) runif(starts, x[1], x[2]))   # one matrix for all, as psp.R

one <- function(i) {
  train <- participant_training(i)
  .evaluate <- function(x) evaluate(params = x, stimuli = stimuli, tr = train)
  set.seed(2026)
  seconds <- system.time(
    res <- pspGlobal(model = .evaluate, discretize = discretize_na,
      control = list(radius = 1, lower = lower, upper = upper, init = jump, dimensionality = 5, responses = 5,
                     parameter_names = c("P", "alpha", "mu", "rho", "beta"), stimuli_names = stimuli,
                     population = 1000000, iterations = iterations),
      quiet = TRUE)
  )[["elapsed"]]
  n <- dim(res$ordinal_patterns)[3]
  list(summary = data.frame(ppt = i, seconds = seconds, iterations = res$iterations, patterns = n,
                            evaluations = sum(res$ordinal_counts)),
       patterns = data.frame(ppt = i, count = as.vector(res$ordinal_counts),
                             key = apply(res$ordinal_patterns, 3, function(m) paste(m[upper.tri(m)], collapse = ","))))
}
wall <- system.time(results <- mclapply(ppts, one, mc.cores = workers, mc.preschedule = FALSE))[["elapsed"]]
failed <- vapply(results, inherits, logical(1), "try-error")
if (any(failed)) stop(results[[which(failed)[1]]])
write.csv(do.call(rbind, lapply(results, `[[`, "summary")), file.path(out, "summary.csv"), row.names = FALSE)
write.csv(do.call(rbind, lapply(results, `[[`, "patterns")), file.path(out, "patterns.csv"), row.names = FALSE)
write.csv(data.frame(workers = workers, participants = n_ppt, wall_seconds = wall), file.path(out, "wall.csv"),
          row.names = FALSE)
cat("done in", wall, "s\n")
