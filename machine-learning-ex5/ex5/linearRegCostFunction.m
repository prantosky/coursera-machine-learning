function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Compute the predictions
h_x = X * theta;

% Create a new array with no regulaization for the bias term
theta_new = [0 ; theta(2:end)];

% Compute the cost
J = (1/(2*m)) .* ((h_x - y)' * (h_x - y)) + (lambda/(2*m)) .* (theta_new' * theta_new);

% Compute the gradients
grad =  (1/m) .* (X' * (h_x - y))   + (lambda/m) .* theta_new;


% =========================================================================

grad = grad(:);

end
