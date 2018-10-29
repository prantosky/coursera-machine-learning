function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Adding bias to each datapoint in first/input layer
a1 = [ones(m,1) X];

z1 = a1 * Theta1';

% Adding bias to each datapoint in second layer
a2 = [ones(size(z1),1) sigmoid(z1)];

z2 = a2 * Theta2'

% Computing the output of third layer
a3 = sigmoid(z2);

% Getting the precidicted label by taking the max score of of the labels
[max_val , p] =max(a3,[],2);

% Changing label '10' as '0'
% p = mod(p,10);

% =========================================================================


end
