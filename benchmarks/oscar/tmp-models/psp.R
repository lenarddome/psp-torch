#' @title pspNEW
#' @description
#' Run parameter space partitioning for each participant in the supplied data
#' and save both the explored parameter space and ordinal pattern summaries to
#' disk.
#'
#' @param iteration Number of partitioning iterations passed to pspGlobal.
#' @param population Population size used by the partitioning routine.
#' @param data Participant-level dataset containing at least ppt, abstim,
#'   phase, and trial columns.
#' @param stimuli Stimulus labels to evaluate in the NNRAS model.
#' @param init Initial parameter vector kept for compatibility with older calls.
#'   The current implementation samples starting points from lower and upper.
#' @param lower Lower bounds for each fitted parameter.
#' @param upper Upper bounds for each fitted parameter.
#' @param hypersphere Radius argument forwarded to pspGlobal.
#' @param pnames Names of the fitted parameters, used in saved outputs.
#' @param export_packages Packages that must be loaded on each parallel worker.
#' @param path Directory where participant-specific CSV and RData files are
#'   written.
#'
#' @return No explicit return value. The function writes participant-level PSP
#'   outputs to disk as a side effect.
pspNEW <- function(iteration = 20000, population = 1000000,
                   data = NULL, stimuli = c("A", "B", "BC", "C"),
                   init = c(3, 0.59, 0.35, 2, 3),
                   lower = c(1, 1e-3, 1e-3, 1e-3, 1e-3),
                   upper = c(5, 1, 1, 5, 5),
                   hypersphere = 1,
                   pnames = c("P", "alpha", "mu", "rho", "beta"),
                   export_packages = c("data.table", "clobe", "sunny", "psp"),
                   path = ".") {
    if (is.null(data)) {
        stop("No data provided. Please supply a participant-level dataset with at least ppt, abstim, phase, and trial columns.")
    }
    ## reserve cores for other work on the machine while running each participant in parallel
    core_numbers <- detectCores() - 2
    cl <- makeCluster(core_numbers)
    registerDoParallel(cl)

    print(paste("Using", core_numbers, "cores for parallel processing."))

    jump <- apply(rbind(lower, upper), 2, function(x) {
        dis <- runif(1000, x[1], x[2])
        return(dis)
    })

    start <- Sys.time()
    ## print start time
    print(paste("Simulation began at", start))

    ## run a separate PSP exploration for each participant's trial sequence
    foreach(
        i = unique(data$ppt),
        .packages = export_packages,
        .export = c("ConvertToStateList", "evaluate", "discretize"),
        .combine = "rbind"
    ) %dopar% {
        if (!("data.table" %in% is(data))) data <- data.table(data)
        participant <- data[ppt == i]
        tracking <- data.table(ppt = i, started = as.POSIXct(Sys.time()))

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
        train <- ConvertToStateList(
            schedule = trial_order,
            block = phase_index,
            trials = trials,
            ppt = i,
            design = mode
        )

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
            quiet = TRUE,
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
        ## save tracking information to monitor progress and identify any participants that may be causing issues in the exploration routine, and save the updated tracking information to disk
        tracking <- tracking[, completed := Sys.time()]
        tracking <- tracking[, explored := TRUE]
        fwrite(tracking, paste(path, "/tracking.csv", sep = ""), append = TRUE)
        ## save the frequency counts of each ordinal pattern for later analyses
        frequencies <- data.table(
            ppt = i,
            dataset = unique(participant$dataset),
            id = seq_along(out$ordinal_counts),
            frequency = t(out$ordinal_counts),
            dataset = unique(participant$dataset)
        )

        fwrite(frequencies, paste(path, "/frequencies.csv", sep = ""), append = TRUE)
        rm(partitions) ## clean up workspace
        gc()
        message <- paste("Completed participant ", i, " of ", length(unique(data$ppt)), sep = "")
        print(message)
    }

    ## close the parallel backend once all participant jobs have finished
    stopImplicitCluster()
    end <- Sys.time()
    print(paste("Simulation completed at", end))
    print(paste("Total runtime:", difftime(end, start, units = "mins"), "minutes"))
}
