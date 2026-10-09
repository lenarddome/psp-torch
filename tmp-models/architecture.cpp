// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "architecture.h"

using namespace Rcpp;
using namespace arma;

/**
 * Attentional Gating - Calculate normalized attention gain
 * @brief Applies attentional gating to the input vector using element-wise modulation and p-norm normalization.
 *
 * @param input The input row vector to be gated. These are external excitations or stimuli vector.
 * @param eta The attention modulation row vector. These are the attention weights (underlying saliences) or gating factors.
 * @param P The exponent used for p-norm normalization.
 * @return List. A list containing the gated output, attention gain, and the p-norm of the gain.
 */
Rcpp::List AttentionalGating(arma::rowvec input, arma::rowvec eta, double P)
{
    List results;
    // Calculate the attention gain
    rowvec attention_gain = input % eta;
    // calculate the vector norm
    rowvec gain_p(attention_gain);
    for (uword j = 0; j < attention_gain.n_elem; ++j)
    {
        gain_p[j] = pow(attention_gain[j], P);
    }
    double p_sum = pow(sum(gain_p), 1 / P);
    // Normalize the attention gain with the p-norm
    rowvec gated_output = attention_gain / p_sum;
    results = List::create(
        Rcpp::Named("gated_output") = gated_output,
        Rcpp::Named("gain_norm") = p_sum);
    return results;
}

/**
 * ConnectionWeights - Calculate connection weights in a fully connected two-layer network based on input and attention modulation
 * @brief Computes the connection weights by applying attentional gating to the input vector in a two-layer fully connected network.
 * @param W mat. The weight matrix of the connections in the network, where each node in the first layer is connected to every node in the second layer. The column represents nodes in the first layer and the row represents nodes in the second layer.
 * @param attention rowvec. The gated attention row vector, representing the gated external excitations or stimuli vector.
 * @param output colvec. The output vector representing the output units of the network that are eligible to receive input from the first layer.
 * @return mat The connection weights matrix after applying attentional gating.
 */
arma::mat ConnectionWeights(arma::mat W, arma::rowvec attention, arma::colvec output)
{
    // Calculate the connection weights by multiplying the weight matrix with the gated attention vector
    mat connection_weights = W.each_row() % attention;
    connection_weights.each_col() %= output; // modulate by the output vector
    // set diagonal elements to 0
    connection_weights.diag().zeros();
    return connection_weights;
}

/**
 * ConnectionWeights - Calculate connection weights in a fully connected two-layer network based on input and attention modulation
 * @brief Computes the connection weights by applying attentional gating to the input vector in a two-layer fully connected network.
 * @param W mat. The weight matrix of the connections in the network, where each node in the first layer is connected to every node in the second layer. The column represents nodes in the first layer and the row represents nodes in the second layer.
 * @param attention mat. The gated attention matrix, representing the gated external excitations or stimuli matrix.
 * @param output colvec. The output vector representing the output units of the network that are eligible to receive input from the first layer.
 * @return mat The connection weights matrix after applying attentional gating.
 */
arma::mat ConnectionWeightsFull(arma::mat W, arma::mat attention, arma::colvec output)
{
    // Calculate the connection weights by multiplying the weight matrix with the gated attention matrix
    mat connection_weights = W % attention;
    connection_weights.each_col() %= output; // modulate by the output vector
    // set diagonal elements to 0
    connection_weights.diag().zeros();
    return connection_weights;
}

/**
 * ThurstonianChoice - Calculate the Thurstonian choice model output
 * @brief Computes the Thurstonian choice model output based on the connection weights and a threshold. This decision process is a winner-take-all process, where the model selects the most activated output after repeated excitation and decay of output units, maximum 100 iterations.
 * @param connection_weights mat. The connection weights matrix, where each column represents a node in the first layer and each row represents a node in the second layer.
 * @param output colvec. The output vector representing the current state of the network, where each element corresponds to a node in the second layer. This vector tells the model what output nodes can be active.
 * @param tau double. The threshold value used to determine the choice output.
 * @param E double. Excitation parameter.
 * @param D decay. Decay parameter.
 * @return List. A list containing the choice output vector at the time of the decision, the winner node index, and the time of the decision.
 */
Rcpp::List ThurstonianChoice(arma::mat connection_weights, arma::colvec output, double tau, double E, double D, double sigma)
{
    // initialize variables
    int time_of_decision = 0;
    // Sum the connection weights across rows (i.e., for each output node, sum all inputs from the first layer)
    colvec new_output = sum(connection_weights, 1) % output;
    uword winner_index = 0;
    List result;
    // iterate until a winner is found or maximum iterations reached
    for (int t = 0; t < 100; ++t)
    {
        // calculate the new output by applying the connection weights to the current output
        // rowsum is used to sum the contributions from all nodes in the first layer to each node in the second layer
        colvec total_input(output.n_elem, fill::zeros); // output activations
        colvec activation(output.n_elem, fill::zeros);  // total input to the output nodes
        for (uword i = 0; i < total_input.n_elem; ++i)
        {
            double noise = arma::randn(arma::distr_param(0.0, sigma)); // random guassian noise
            double sum_others = 0.0;                                   // summed activation of all other elements
            for (uword j = 0; j < new_output.n_elem; ++j)
            {
                if (j != i)
                {
                    sum_others += new_output[j] * output[j];
                }
            }
            // calculate the totla input to the output node i
            total_input[i] = (noise + new_output[i]) - sum_others;
            // save the original activation
            activation[i] = new_output[i] * output[i];
        }
        for (uword i = 0; i < new_output.n_elem; ++i)
        {
            // calcualte sigma - magnitude
            new_output[i] = ((activation[i] + E * total_input[i]) / (1 + E * total_input[i] + D)) * output[i];
        }
        // clamp new_output to be non-negative
        new_output = clamp(new_output, 0.0, datum::inf);
        // get the index of the two maximum activations in new_output
        uvec sorted_indices = sort_index(new_output, "descend");
        uword max_index = sorted_indices[0];        // index of the maximum activation
        uword second_max_index = sorted_indices[1]; // index of the second maximum activation
        // check if the difference between the two maximum activations is greater than the threshold tau
        if (new_output[max_index] - new_output[second_max_index] > tau)
        {
            winner_index = max_index; // found a winner
            time_of_decision = t + 1; // time of decision is the current iteration + 1
            break;
        }
        if (t == 99)
        {
            winner_index = max_index; // if no winner is found, take the maximum index
            time_of_decision = t + 1; // time of decision is the maximum iterations
        }
    }
    result = List::create(
        Named("last_output_activations") = new_output,
        Named("winner_index") = winner_index,
        Named("time_of_decision") = time_of_decision);
    return result;
}

arma::colvec SoftMax(arma::colvec activation, arma::colvec output, double beta)
{
    colvec scaled = activation * beta;
    colvec power(activation.n_elem, fill::zeros);
    for (uword i = 0; i < output.n_elem; ++i)
    {
        power[i] = std::exp(scaled[i]) * output[i];
    }
    colvec probabilities = (power / sum(power));
    return probabilities;
}

/**
 * GetTeacher - Determines the teacher output based on the current mode of the network.
 *
 * @brief Returns the appropriate teacher signal for learning, depending on the mode:
 *   1 - Observational learning, no responding, reset model to initial state
 *   0 - Observational learning, no responding
 *   2 - No learning, only responding
 *   3 - Supervised learning, make response and reset network
 *   4 - Supervised learning, make response
 *   5 - Unsupervised learning, make response
 *
 * @param output colvec  The currently active output units.
 * @param stimuli rowvec The currently active input units.
 * @param mode   int     The mode of the network (see above).
 * @return colvec        The teacher signal for the current trial.
 */
arma::colvec GetTeacher(arma::colvec feedback, arma::rowvec stimuli, int mode)
{
    if (mode == 0 || mode == 1 || mode == 5)
    {
        // unsupervised learning, return the output as the teacher
        return stimuli.as_col();
    }
    if (mode == 3 || mode == 4)
    {
        // supervised learning, return the stimuli as the teacher
        return stimuli.as_col() + feedback.as_col();
    }
    else
    {
        // if mode is not recognized, raise error
        Rcpp::stop("Mode not recognized. Use 0 for unsupervised learning or 3/4 for supervised learning.");
    }
}

// NOTE: not used in the model
/**
 * HumbleTeacher - Calculate the humble teacher output
 * @brief Computes the humble teacher (Kruschke, 1992) output based on the current output and the teacher's output. It is essentially a ReLU-like modulation of the teacher's output towards the current output.
 * @param output colvec. The current output vector of the model.
 * @param teacher colvec. The teacher's output vector, representing the desired or target output.
 * @return vec. The prediction error calculated as a difference between summed output node activations and the humble teacher.
 */
Rcpp::List HumbleTeacher(arma::colvec activations, arma::colvec teacher, arma::colvec output)
{
    // initialize the humble teacher output vector
    vec humble(output.n_elem, fill::zeros);
    List results;
    // Calculate the humble teacher output
    for (uword i = 0; i < output.n_elem; ++i)
    {
        // if teacher ==1 and if activation is larger than the teacher's output, substitute the teacher's output
        if (teacher[i] == 1 && activations[i] > teacher[i])
        {
            humble[i] = activations[i];
        }
        else
        {
            humble[i] = teacher[i];
        }
    }
    // calculate prediction error
    vec prediction_error = (humble - activations) % output;
    results = List::create(
        Rcpp::Named("humble_teacher_output") = humble,
        Rcpp::Named("prediction_error") = prediction_error);
    return results;
}

/**
 * Update connection weights with the humble teacher output
 * @brief Updates the connection weights based on the humble teacher output.
 * @param connection_weights mat. The current connection weights matrix.
 * @param error vec. The humble teacher output vector.
 * @param input rowvec. The input vector, representing the currently active nodes.
 * @param output colvec. The output vector, representing the currently active output nodes.
 * @param gain rowvec. The attention gain vector, representing the attention modulation for each input node.
 * @param alpha double. The learning rate for updating the connection weights.
 * @return mat. The updated connection weights matrix.
 * @details This function essentially the delta rule for updating the connection weights in a neural network.
 */
Rcpp::List DeltaRule(arma::mat connection_weights, arma::colvec error, arma::rowvec input, arma::rowvec gain, double alpha)
{
    // create copy of connection weights
    mat updated_weights(connection_weights);
    mat delta(connection_weights.n_rows, connection_weights.n_cols, fill::zeros);
    List results;
    // Update the connection weights using the delta rule
    for (uword i = 0; i < connection_weights.n_rows; ++i)
    {
        for (uword j = 0; j < connection_weights.n_cols; ++j)
        {
            // update the connection weight
            delta(i, j) += alpha * error[i] * (input[j] * gain[j]);
            updated_weights(i, j) += delta(i, j);
        }
    }
    // ensure the diagonal remains 0
    updated_weights.diag().zeros();
    delta.diag().zeros();
    // return the updated connection weights
    results = List::create(
        Rcpp::Named("updated_weights") = updated_weights,
        Rcpp::Named("delta") = delta);
    return results;
}

/**
 * Update connection weights with the humble teacher output
 * @brief Updates the connection weights based on the humble teacher output.
 * @param connection_weights mat. The current connection weights matrix.
 * @param error vec. The humble teacher output vector.
 * @param input rowvec. The input vector, representing the currently active nodes.
 * @param output colvec. The output vector, representing the currently active output nodes.
 * @param gain rowvec. The attention gain vector, representing the attention modulation for each input node.
 * @param alpha double. The learning rate for updating the connection weights.
 * @return mat. The updated connection weights matrix.
 * @details This function essentially the delta rule for updating the connection weights in a neural network.
 */
Rcpp::List DeltaRuleFull(arma::mat connection_weights, arma::colvec error, arma::rowvec input, arma::mat gain, double alpha)
{
    // create copy of connection weights
    mat updated_weights(connection_weights);
    mat delta(connection_weights.n_rows, connection_weights.n_cols, fill::zeros);
    List results;
    // Update the connection weights using the delta rule
    for (uword i = 0; i < connection_weights.n_rows; ++i)
    {
        for (uword j = 0; j < connection_weights.n_cols; ++j)
        {
            // update the connection weight
            delta(i, j) += alpha * error[i] * (input[j] * gain(i, j));
            updated_weights(i, j) += delta(i, j);
        }
    }
    // ensure the diagonal remains 0
    updated_weights.diag().zeros();
    delta.diag().zeros();
    // return the updated connection weights
    results = List::create(
        Rcpp::Named("updated_weights") = updated_weights,
        Rcpp::Named("delta") = delta);
    return results;
}

/**
 * AttentionShift - Calculate the attention shift according to gradient descent on error.
 * @brief Computes the attention shift based on the connection weights, input vector, output vector, error vector, and attention vector.
 * @param weights mat. The connection weights matrix.
 * @param input rowvec. The input vector representing the currently active nodes.
 * @param error colvec. The error vector representing the prediction error.
 * @param gain rowvec. The attention vector representing the current attention gains.
 * @param pnorm double. The p-norm value used for normalization.
 * @param P double. The exponent used for p-norm normalization.
 * @param rho double. The scaling factor for the salience calculation.
 * @return rowvec. The salience vector representing the attention shift for each input node.
 */
arma::rowvec AttentionShift(arma::mat weights, arma::colvec predictions, arma::rowvec input,
                            arma::colvec error, arma::rowvec gain, double pnorm, double P, double rho)
{
    rowvec salience(gain);                                                // salience vector
    mat attended_activations(weights.n_rows, weights.n_cols, fill::ones); // creaate a matrix to hold intermediate computations

    // calculate model predictions
    mat activations = weights.each_row() % input; // apply attention gain to weights

    // raise salience to power of P-1
    for (uword j = 0; j < gain.n_elem; ++j)
    {
        salience[j] = pow(gain[j], P - 1) * input[j];
    }

    // calculate the attended activations
    attended_activations.each_col() %= predictions; // modulate by prediction error
    attended_activations.each_row() %= salience;    // modulate by input
    attended_activations.diag().zeros();           // set diagonal to 0

    mat out = (activations - attended_activations); // calculate the difference between activations and attended activations
    out = out.each_col() % error;                   // modulate by the error
    mat out_sum = sum(out, 0);                      // sum across rows
    // TODO: mute stimuli not present
    mat delta_eta = arma::tanh(rho * pow(pnorm, -1) * out_sum % input); // calculate the salience update

    return delta_eta; // calculate the error for each input node
}

/**
 * AttentionUpdate - Update the attention vector based on the salience vector according to the delta rule.
 * @brief Updates the attention vector based on the portion of the recurrent gain.
 * @param attention_gain rowvec. The current attention gain vector.
 * @param recurrent_gain rowvec. The attention gain after the attention shift has been completed.
 *
 */
arma::rowvec AttentionUpdate(arma::rowvec attention_gain, arma::rowvec recurrent_gain, arma::rowvec input, double mu)
{
    // Update the attention vector using the delta rule
    rowvec updated_attention = mu * (recurrent_gain - attention_gain);
    for (uword j = 0; j < input.n_elem; ++j)
    {
        if (input[j] == 0)
        {
            updated_attention[j] = 0; // if the input is 0, set the attention update to 0
        }
    }
    // clamp the attention vector to be non-negative
    return updated_attention;
}
