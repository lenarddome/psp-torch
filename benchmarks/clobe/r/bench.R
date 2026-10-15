# Seconds per call of the functions of the R package clobe on the workloads of
# benchmarks/clobe/run.py, with the inputs it writes.
#
# Usage: Rscript bench.R <library with clobe 0.8.0> <inputs directory> <output csv>

args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], .libPaths()))
library(clobe)
stopifnot(packageVersion("clobe") == "0.8.0")
inputs <- args[2]
out_file <- args[3]

data <- as.matrix(read.csv(file.path(inputs, "data.csv")))
dimnames(data) <- NULL
human_idx <- read.csv(file.path(inputs, "human_idx.csv"))$index + 1
model_idx <- read.csv(file.path(inputs, "model_idx.csv"))$index + 1
x_idx <- read.csv(file.path(inputs, "x_idx.csv"))$index + 1
thresholds <- c(0.1, -0.1)
six <- orderings(6)$inequality_matrices
upper_sum <- function(m) sum(m[upper.tri(m)])

# median seconds per call: one warm-up call, then samples of at least 0.05 s
# each, up to 5 samples or 30 s
measure <- function(fn) {
  start <- proc.time()[["elapsed"]]
  check <- fn()
  once <- max(proc.time()[["elapsed"]] - start, 1e-6)
  inner <- max(1, ceiling(0.05 / once))
  samples <- numeric(0)
  total <- 0
  while (length(samples) < 5 && (length(samples) < 1 || total < 30)) {
    start <- proc.time()[["elapsed"]]
    for (i in seq_len(inner)) fn()
    elapsed <- proc.time()[["elapsed"]] - start
    samples <- c(samples, elapsed / inner)
    total <- total + elapsed
  }
  list(seconds = median(samples), samples = length(samples), check = check)
}

rows <- list()
record <- function(fun, size, fn) {
  cat(sprintf("%s %s ... ", fun, size))
  m <- measure(fn)
  cat(sprintf("%.6g s\n", m$seconds))
  rows[[length(rows) + 1]] <<- data.frame(
    "function" = fun, size = size, build = "clobe 0.8.0 (R)",
    seconds = sprintf("%.6g", m$seconds), samples = m$samples,
    check = sprintf("%.15g", m$check), check.names = FALSE)
}

record("imac (one vector)", 1, function() upper_sum(imac(data[1, ], thresholds)))
for (n in c(1e3, 1e4, 1e5)) {
  record("imac (every row)", n, function() {
    sum(vapply(seq_len(n), function(i) upper_sum(imac(data[i, ], thresholds)),
               numeric(1)))
  })
  record("patterns", n, function() patterns(data[seq_len(n), ], thresholds)$n)
}
human <- six[, , human_idx]
for (m in c(1000, 10000, length(model_idx))) {
  model <- six[, , model_idx[seq_len(m)]]
  record("gdistance", m, function() {
    gdistance(human, model, universal = semiorders(6), weight = 0.5,
              frequencies = rep(1 / length(human_idx), length(human_idx)))$gdistance
  })
}
for (n in c(100, 1000, 5000)) {
  x <- six[, , x_idx[seq_len(n)]]
  record("pattern_distance", n, function() sum(pattern_distance(x)))
}
for (k in 5:7) record("orderings (semiorder)", k, function() orderings(k)$n)
for (k in 7:9) record("orderings (weak)", k, function() orderings(k, "weak")$n)
for (k in 5:8) {
  record("brutes", k, function() {
    brutes((0:10) / 10, k, thresholds, return_matrices = k < 8)$n
  })
}

write.csv(do.call(rbind, rows), out_file, row.names = FALSE)
