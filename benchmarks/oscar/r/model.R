# Seconds per evaluation of OSCAR in R + C++, for the first participant: the compiled OSCAR() alone,
# and the routine's evaluate() (fresh random initial state, OSCAR(), then the data.table reduction
# to P(rare) at test), each called once per parameter set as pspGlobal() calls the model.
#
#   Rscript model.R <lib> <tmp-models dir> <out csv> <evaluations> <repeats>
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; models <- args[2]; out <- args[3]
n <- as.integer(args[4]); repeats <- as.integer(args[5])
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))

train <- participant_training(unique(ibre$ppt)[1])
tr <- data.matrix(train)
set.seed(1)
par <- sapply(seq_along(lower), function(k) runif(n, lower[k], upper[k]))

oscar_only <- function(x) {
  eta <- matrix(runif(25, 0, 0.001), 5, 5); diag(eta) <- 0
  st <- list(P = x[1], alpha = x[2], mu = x[3], rho = x[4], beta = x[5], eta = eta,
             weights = matrix(runif(25, 0, 0.001), 5, 5), colskip = 4)
  OSCAR(st = st, tr = tr)
}
rows <- NULL
for (what in c("OSCAR() alone", "evaluate()")) {
  fn <- if (what == "evaluate()") function(x) evaluate(params = x, stimuli = stimuli, tr = train) else oscar_only
  for (j in 1:3) fn(par[j, ])   # warm-up
  for (r in seq_len(repeats)) {
    seconds <- system.time(for (j in seq_len(n)) fn(par[j, ]))[["elapsed"]]
    rows <- rbind(rows, data.frame(what = what, rep = r, evaluations = n, seconds = seconds))
    cat(what, r, seconds, "s\n")
  }
}
write.csv(rows, out, row.names = FALSE)
