## function to produce inequality matrix for EXIT
evaluate <- function(params = c(2.87, 0.12, 0.2, 2, 3),
                     stimuli = c("A", "B", "BC", "C"),
                     tr = NULL, xtdo = FALSE) {
    ## random small initial salience; diagonal is 0 to avoid self-connections
    eta <- matrix(abs(runif(25, min = 0, max = 0.001)), nrow = 5, ncol = 5)
    diag(eta) <- 0
    st <- list(
        "P" = params[1],
        "alpha" = params[2],
        "mu" = params[3],
        "rho" = params[4],
        "beta" = params[5],
        ## random small initial weights
        "eta" = eta,
        "weights" = matrix(abs(runif(25, min = 0, max = 0.001)), nrow = 5, ncol = 5),
        ## number of columns to skip to get to the stimulus features
        "colskip" = 4
    )
    output <- OSCAR(
        st = st,
        tr = data.matrix(tr)
    )

    tout <- output$output[, 4:5]
    tout <- data.frame(cbind(0, 0, tout))
    tout[, 1] <- tr$stim
    tout[, 2] <- tr$ctrl
    colnames(tout) <- c("stimulus", "phase", "common", "rare")
    tout <- data.table(tout)[phase == 2]
    tout <- tout[, -2]

    ## turn wide into long
    wide_out <- aggregate(tout[, 2:ncol(tout)], by = list(tout$stim), mean)
    long_out <- melt(data.table(wide_out), id.vars = "Group.1")
    colnames(long_out) <- c("stimulus", "category", "prob")

    ## sort based on stimuli included in  ineqaulity matrix
    ibre <- long_out[stimulus %in% stimuli]
    ## only include a single category option
    ibre <- ibre[category == "rare"]

    ## make sure prob is numeric and create inequality matrix
    ibre <- rbind(cbind(stimulus = "∅", category = "rare", prob = 0.5), ibre)
    ibre[, prob := as.numeric(prob)]
    if (xtdo) {
        return(tout)
    } else {
        return(ibre$prob)
    }
}

discretize <- function(probabilities, thresholds = c(0.125, -0.125)) {
    out_ineqauality <- imac(probabilities = probabilities, thresholds = thresholds)
    if (any(is.na(out_ineqauality[upper.tri(out_ineqauality, diag = FALSE)]))) {
        out_ineqauality[] <- 99
    }
    return(out_ineqauality)
}
