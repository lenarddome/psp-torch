#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <RcppArmadillo.h>

// Shared function declarations
Rcpp::List AttentionalGating(arma::rowvec input, arma::rowvec eta, double P);
arma::mat ConnectionWeights(arma::mat W, arma::rowvec attention, arma::colvec output);
arma::mat ConnectionWeightsFull(arma::mat W, arma::mat attention, arma::colvec output);
Rcpp::List ThurstonianChoice(arma::mat connection_weights, arma::colvec output, double tau, double E, double D, double sigma);
arma::colvec SoftMax(arma::colvec activation, arma::colvec output, double beta);
arma::colvec GetTeacher(arma::colvec feedback, arma::rowvec stimuli, int mode);
Rcpp::List HumbleTeacher(arma::colvec activations, arma::colvec teacher, arma::colvec output);
Rcpp::List DeltaRule(arma::mat connection_weights, arma::colvec error, arma::rowvec input, arma::rowvec gain, double alpha);
Rcpp::List DeltaRuleFull(arma::mat connection_weights, arma::colvec error, arma::rowvec input, arma::mat gain, double alpha);
arma::rowvec AttentionShift(arma::mat weights, arma::colvec predictions, arma::rowvec input, arma::colvec error, arma::rowvec gain, double pnorm, double P, double rho);
arma::rowvec AttentionUpdate(arma::rowvec attention_gain, arma::rowvec recurrent_gain, arma::rowvec input, double mu);

#endif // COMMON_FUNCTIONS_H
