ConvertToStateList <-
    function(schedule, block, trials, ppt, design = "unsupervised") {
        #' @title ConvertToStateList
        #' @description Converts the experimental design into a format compatible with state list processing.
        #' @param schedule character vector. A vector specifying the order of stimuli presentation.
        #' @param block numeric vector. Numeric value indicating the experimental block (0=training or 1=test). Elements correspond to the schedule.
        #' @param trials numeric vector. Numeric value indicating the trial number. Elements correspond to the schedule.
        #' @param ppt any. Participant identifier.
        #' @param design character. The type of experimental design, either "unsupervised" or "supervised". Default is "observational".
        #' @return Returns a data frame with the experimental design formatted for state list processing. See Wills et al. 2017 for details.
        #' @references Wills, A. J., O'Connell, G., Edmunds, C. E., & Inkster, A. B. (2017). Progress in modeling through distributed collaboration: Concepts, tools and category-learning examples. In Psychology of learning and motivation (Vol. 66, pp. 79-115). Academic Press.


        ## create test items for ibre
        stimuli <- rbind(
            ## A,   B,   C,  X,  Common,  Rare
            c(1, 0, 0, 0, 0), # A
            c(0, 1, 0, 0, 0), # B
            c(0, 0, 1, 0, 0), # C
            c(1, 1, 0, 0, 0), # AB
            c(1, 1, 0, 1, 0), # ABX
            c(1, 0, 1, 0, 0), # AC
            c(1, 0, 1, 0, 1), # ACY
            c(0, 1, 1, 0, 0) # BC
        )

        if (design == "supervised") {
            response <- data.frame(rbind(
                ## A,   B,   C,  X,  Common,  Rare
                c(0, 0, 0, 1, 1), # training
                c(0, 0, 0, 1, 1) # test
            ))
        } else if (design == "unsupervised") {
            response <- data.frame(rbind(
                ## A,   B,   C,  X,  Common,  Rare
                c(0, 0, 0, 0, 0), # training
                c(0, 0, 0, 1, 1) # test
            ))
        } else {
            stop("Unknown design type. Please use 'unsupervised' or 'supervised'.")
        }

        feedback <- data.frame(rbind(
            ## A,   B,   C,  Common,  Rare
            c(0, 0, 0, 1, 0), # common
            c(0, 0, 0, 0, 1), # rare
            c(0, 0, 0, 0, 0) # no feedback
        ))

        feedback <- data.frame(cbind(
            stimuli = c("AB", "AC", "NULL"),
            feedback
        ))

        ## create test items for ice
        stimuli <- data.frame(cbind(0, stimuli))

        ## add stimuli names
        stimuli[, 1] <- c("A", "B", "C", "AB", "ABX", "AC", "ACY", "BC")

        ## construct matriux based on trial order
        bigtr <- NULL
        for (idx in seq_along(schedule)) {
            stim_name <- schedule[idx]
            block_val <- block[idx]
            # Determine response row: 1 for training (block==0), 2 for test (block==1)
            resp_row <- ifelse(block_val == 0, 1, 2)
            stim_row <- stimuli[stimuli$X1 == stim_name, ]
            if (design == "unsupervised") {
                feedback_vector <- feedback[feedback$stimuli == "NULL", -1]
            } else if (design == "supervised") {
                if (stim_name == "AB") {
                    feedback_vector <- feedback[feedback$stimuli == "AB", -1]
                } else if (stim_name == "AC") {
                    feedback_vector <- feedback[feedback$stimuli == "AC", -1]
                } else {
                    feedback_vector <- feedback[feedback$stimuli == "NULL", -1]
                }
                if (block_val == 2) {
                    feedback_vector <- feedback[feedback$stimuli == "NULL", -1]
                }
            } else {
                stop("Unknown design type. Please use 'unsupervised' or 'supervised'.")
            }
            row <- cbind(stim_row, response[resp_row, ], feedback_vector)
            bigtr <- rbind(bigtr, row)
        }

        ## add extra info for slp
        block <- as.numeric(block)
        bigtr <- cbind(ctrl = block, ppt, trials, bigtr)
        if (design == "unsupervised") {
            bigtr[1, 1] <- 1
        } else if (design == "supervised") {
            bigtr[1, 1] <- 3
        } else {
            stop("Unknown design type. Please use 'unsupervised' or 'supervised'.")
        }
        # tidy up and return output
        rownames(bigtr) <- NULL
        colnames(bigtr) <- c(
            "ctrl", "ppt", "trials", "stim",
            "x1", "x2", "x3", "x4",
            "x5", "y1", "y2", "y3", "y4", "y5",
            "f1", "f2", "f3", "f4", "f5"
        )
        return(bigtr)
    }

FillInMissingData <- function(data) {
    #' @title FillInMissingData
    #' @description Fills in missing data for each response categories in aggregate results for each participant.
    #' @param data data.table. The data frame containing the experimental design.
    #' @details This function ensures that all combinations of participant, stimulus, and response are represented in the data frame, filling in missing combinations with zeros. For example, if participant responded with Category 1 to stimulus A, and had zero responses with Category 2, this function will add a row for Category 2 with a count of zero.
    #' @return Returns a data frame with missing data filled in.

    stimuli <- unique(data$abstim)
    responses <- unique(data$abresp)
    ## create empty data.table to fill in missing combinations
    filled_data <- data.table(
        abstim = rep(stimuli, each = 2),
        abresp = rep(c("common", "rare"), times = length(stimuli)),
        N = 0
    )
    for (i in seq_len(nrow(data))) {
        stim <- data$abstim[i]
        resp <- data$abresp[i]
        if (resp == 0) {
            resp <- "common"
        } else if (resp == 1) {
            resp <- "rare"
        }
        count <- data$N[i]
        filled_data[abstim == stim & abresp == resp, N := count]
    }
    filled_data <- filled_data[order(abstim, abresp)]
    return(filled_data)
}
