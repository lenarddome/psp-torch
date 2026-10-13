// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "architecture.h"

using namespace Rcpp;
using namespace arma;

/**
 * The function that applies the complete model to data.
 * ctr column codes:
 * 1 - observational learning, no responding, reset the model to initial state
 * 0 - observational learning, no responding
 * 2 - no learning, only responding
 * 3 - supervised learning, make response in addition to learning, reset network
 * 4 - supervised learning, make a response in addition to learning
 * 5 - unsupervised learning, make a response in addition to learning
 */
// [[Rcpp::export]]
List OSCAR(List st, arma::mat tr)
{
    double P = as<double>(st["P"]);         // attentional normalization
    double alpha = as<double>(st["alpha"]); // learning rate parameter
    double mu = as<double>(st["mu"]);       // attention update rate
    double rho = as<double>(st["rho"]);     // attentional shift rate
    double beta = as<double>(st["beta"]);   // softmax temperature
    int colskip = as<int>(st["colskip"]);

    mat iweights = as<mat>(st["weights"]); // initial connection weights
    // set diagonal to 0
    iweights.diag().zeros();

    mat ieta = as<mat>(st["eta"]); // initial salience
    // set diagonal to 0
    ieta.diag().zeros();

    // declare variables that will be updated through the sim
    mat weights(iweights);
    mat eta(ieta);                                    // attention matrix that will be updated through the sim
    mat gated(ieta.n_rows, ieta.n_cols, fill::zeros); // create a matrix to hold the gated attention vector for each trial
    mat out;
    // get size of weight matrix
    uword m = weights.n_rows, n = weights.n_cols;
    // get size of tr
    uword trow = tr.n_rows, tcol = tr.n_cols;
    // store intermediary variables
    rowvec train(tcol); // vector of current trial
    rowvec input(m);
    colvec output(n, fill::zeros);   // output activations for the Choice Competition
    colvec feedback(n, fill::zeros); // feedback vector
    // store outputs
    cube connections(m, n, trow, fill::zeros);    // store trial by trial connection weights
    cube delta_salience(m, n, trow, fill::zeros); // salience updates for the Attention Shift
    cube attention(m, n, trow, fill::zeros);      // gated attention vector for the Attentional Competition

    mat activations(trow, m, fill::zeros); // output activations
    mat error(trow, n, fill::zeros);       // store the prediction error for each trial
    mat prob(trow, m, fill::zeros);
    List outFIN;

    // run model
    for (uword i = 0; i < trow; i++)
    {
        // reset network to inital state if ctrl = 1
        if (tr(i, 0) == 1 || tr(i, 0) == 3)
        {
            // set weights to initial state from st
            weights.zeros();
            weights += iweights;
            // set salience to initial state from st
            eta.zeros();
            eta += ieta;
        }
        // Store the connection weights
        connections.slice(i) = weights;
        // Extract current trial
        train = tr.row(i);
        // Extract input node activations
        input = train.subvec(colskip, colskip + n - 1).as_row();
        // extract output node activations
        output = train.subvec(n + colskip, colskip + 2 * n - 1).as_col();
        // if output contains only 0s, set all to 1
        if (sum(output) == 0)
        {
            output.fill(1);
        }
        // Extract teaching signals
        feedback = train.subvec(tcol - n, tcol - 1).as_col();
        if (sum(feedback) == 0)
        {
            feedback = input.as_col();
        }
        // Equation 1: Attentional Gating
        // rowvise application of attentional gating to allow for different attention vectors for each output node
        for (uword k = 0; k < m; ++k)
        {
            List gated_row = AttentionalGating(input, eta.row(k), P);
            gated.row(k) = as<rowvec>(gated_row["gated_output"]);
        }
        // Equation 2: Connection Weights
        mat connection_weights = ConnectionWeightsFull(weights, gated, output);
        // Make a choice based on the connection weights
        colvec activation = sum(connection_weights, 1); // sum the connection weights across rows
        activations.row(i) = activation.t();            // store the activations
        colvec probabilities = SoftMax(activation, output, beta);
        prob.row(i) = probabilities.t();

        if (tr(i, 0) != 2)
        {

            output.fill(1); // set output to all 1s for learning
            // output.zeros();
            colvec teacher = GetTeacher(feedback, input, tr(i, 0));
            // output[n - 1] = 1;
            // output[n - 2] = 1;
            rowvec complete_input = teacher.as_row();
            // rowvec complete_input = input;
            colvec prediction_error_prime(teacher.n_elem, fill::zeros); // prediction error for the humble teacher
            mat gated_prime(gated);                                     // make a copy of the gated output
            mat eta_prime(eta);                                         // make a copy of the saliences
            // Outcome-specific application of attention shifts
            for (uword k = 0; k < m; ++k)
            {
                rowvec eta_prime_row = eta_prime.row(k);            // make a copy of the saliences for this iteration
                rowvec prediction_row = prediction_error_prime.t(); // make a copy of the prediction error for this iteration
                double teacher_row = teacher(k);                    // make a copy of the teacher for this iteration
                // Recurrent attention shifts
                for (uword j = 0; j < 10; ++j)
                {
                    List attention_gates_prime = AttentionalGating(complete_input, eta_prime_row, P); // recalculate the attention gain
                    gated_prime = as<rowvec>(attention_gates_prime["gated_output"]);                  // get the gated output
                    double pnorm = as<double>(attention_gates_prime["gain_norm"]);                    // get the p-norm of the gain
                    mat connection_weights_prime = weights.row(k) % gated_prime * output(k);          // recalculate connection weights with the updated gated output
                    colvec prediction_prime = sum(connection_weights_prime, 1);                       // sum the connection weights across rows
                    prediction_error_prime = (teacher_row - prediction_prime);                        // calculate the prediction error
                    rowvec delta_eta = AttentionShift(weights.row(k), prediction_prime, complete_input, prediction_error_prime, gated_prime, pnorm, P, rho);
                    eta_prime_row += delta_eta;
                    eta_prime_row.clamp(1e-3, datum::inf); // clamp the attention vector to be non-negative
                }
                eta.row(k) += AttentionUpdate(eta.row(k), eta_prime_row, complete_input, mu); // update the attention vector
                eta.row(k).clamp(1e-3, datum::inf);                                           // clamp the attention vector to be non-negative
            }
            delta_salience.slice(i) = eta; // store the salience map
            // recompute prediction error with the final attention map

            for (uword k = 0; k < m; ++k)
            {
                List gated_row = AttentionalGating(complete_input, eta.row(k), P);
                gated.row(k) = as<rowvec>(gated_row["gated_output"]);
            } // recalculate the attention gain
            mat connection_weights_prime = ConnectionWeightsFull(weights, gated, output);                // recalculate connection weights with the updated gated output
            colvec weight_sums = sum(connection_weights_prime, 1);                                       // sum the connection weights across rows
            prediction_error_prime = (teacher - weight_sums);                                            // calculate the prediction error
            List updates = DeltaRuleFull(weights, prediction_error_prime, complete_input, gated, alpha); // update the connection weights
            weights = as<mat>(updates["updated_weights"]);                                               // get the updated connection weights
            error.row(i) = prediction_error_prime.as_row();                                              // store the prediction error
        }
    }
    outFIN = List::create(
        Rcpp::Named("connections") = connections,
        Rcpp::Named("activations") = activations,
        Rcpp::Named("salience") = delta_salience,
        Rcpp::Named("error") = error,
        Rcpp::Named("output") = prob,
        Rcpp::Named("weights") = weights,
        Rcpp::Named("eta") = eta);

    return outFIN;
}
