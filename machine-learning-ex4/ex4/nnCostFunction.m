function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

vec_y = zeros(size(y),num_labels);

ind = [ 1:size(y) ; y']';
linear_ind = sub2ind([size(vec_y)],ind(:,1),ind(:,2));
vec_y = vec_y(:);
vec_y(linear_ind) = 1;
y = reshape(vec_y,size(y),num_labels);

% Forward Propogation
a1 = [ones(m,1) X];  % size(a1) = 5000 401

z1 = a1 * Theta1';   % size(z1) = 5000 25

a2 = [ones(size(z1),1) sigmoid(z1)];   % size(a2) = 5000 26

z2 = a2 * Theta2';   % size(z2) = 5000 10

a3 = sigmoid(z2);   % size(a3) = 5000 10

% Cost Function

Theta1_new = Theta1(:,2:end);
Theta2_new = Theta2(:,2:end);

J = (1/m) .* sum(sum((-y .* log(a3) - (1-y) .* log(1 - a3)),2)) ...
		+ (lambda/(2*m)) .* (sum(sum(Theta1_new .^ 2, 2)) + sum(sum(Theta2_new .^ 2, 2)));


% Calculating the gradients

g_prime = sigmoidGradient(z1);


del_3 = a3 - y;  % size(del_3) = 5000 10

del_2 = (del_3 * Theta2_new) .* g_prime;  % size(del_2) = 5000 26

Theta2_grad = Theta2_grad + (1/m) .* del_3' * a2 + lambda/m .* [zeros(size(Theta2_new),1) Theta2_new];

Theta1_grad = Theta1_grad + (1/m) .* del_2' * a1 + lambda/m .* [zeros(size(Theta1_new),1) Theta1_new];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
