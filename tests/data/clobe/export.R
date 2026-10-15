# Exports the reference outputs of the R package clobe 0.8.0 that tests/test_clobe.py checks against.
# Run from the psp-torch root: RLIB=<library with clobe 0.8.0> CLOBE=<clobe source> OUT=tests/data/clobe Rscript tests/data/clobe/export.R
.libPaths(c(Sys.getenv("RLIB"), .libPaths()))
library(clobe)
stopifnot(packageVersion("clobe") == "0.8.0")
src <- Sys.getenv("CLOBE"); out <- Sys.getenv("OUT")
# a cube (k, k, n) as rows of n, the strict upper triangle read row by row
flat <- function(x) {
  k <- dim(x)[1]; idx <- which(upper.tri(matrix(0, k, k)), arr.ind = TRUE)
  idx <- idx[order(idx[, 1], idx[, 2]), , drop = FALSE]
  t(apply(x, 3, function(m) m[idx]))
}
write <- function(x, name) write.csv(x, file.path(out, name), row.names = FALSE)
# imac fixture of the R test suite
load(file.path(src, "tests/testthat/data/test_imac.RData"))
set.seed(1); p <- runif(15)
stopifnot(isTRUE(all.equal(stored, imac(p, c(0.1, -0.1)))))
write(data.frame(value = sprintf("%.17g", p)), "imac_values.csv")
write(stored, "imac_stored.csv")
# gdistance fixture of the R test suite
load(file.path(src, "tests/testthat/data/test_gdistance.RData"))
write(flat(predicted), "gdistance_predicted.csv")
write(flat(discovered), "gdistance_discovered.csv")
# random results: imac on both orderings, and g-distance between their patterns
set.seed(2026)
human <- matrix(round(runif(200 * 4), 2), 200)
model <- matrix(round(runif(300 * 4), 2), 300)
write(human, "random_human.csv"); write(model, "random_model.csv")
res <- NULL
for (ordering in c("semiorder", "weak")) {
  h <- patterns(human, c(0.1, -0.1), ordering); m <- patterns(model, c(0.1, -0.1), ordering)
  write(flat(h$inequality_matrices), sprintf("random_human_%s.csv", ordering))
  write(data.frame(pattern = h$pattern), sprintf("random_human_%s_index.csv", ordering))
  u <- if (ordering == "weak") fubini(4) else semiorders(4)
  for (w in c(0, 0.3, 1)) {
    g <- gdistance(h$inequality_matrices, m$inequality_matrices, u, w, h$frequencies, TRUE)
    res <- rbind(res, data.frame(ordering = ordering, weight = w,
      gdistance = sprintf("%.17g", g$gdistance), alpha = sprintf("%.17g", g$alpha),
      beta = sprintf("%.17g", g$beta), accommodation = paste(g$accommodation - 1, collapse = " ")))
  }
}
write(res, "random_gdistance.csv")
# enumerations, in R's order
for (ordering in c("semiorder", "weak")) for (k in 2:5)
  write(flat(orderings(k, ordering)$inequality_matrices), sprintf("orderings_%s_%d.csv", ordering, k))
b <- brutes(seq(0, 1, by = 0.25), 4, c(0.3, -0.3))
write(flat(b$inequality_matrices), "brutes_coarse_4.csv")
