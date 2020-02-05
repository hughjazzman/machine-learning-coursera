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


% Add ones to the X data matrix
X = [ones(m, 1) X];

% K(i) = units in next layer
% K = classes in output layer
% m = training examples
% n = features
% Theta1: rows correspond to hidden unit in next layer (K1 by n + 1)
% X: rows correspond to training examples (m by n+1) (bias unit added)
% X': cols correspond to training examples (n+1 by m)
% Theta* X' will get each row as hidden unit in a2 (K1 by m)
% a2: each row is a hidden unit [similar to a single feature]
% add bias unit of 1 as row vector above all (K1 + 1 by m)
% Theta2: rows correspond to final class in output layer (K by K1 + 1)
% a3: each row is probability of a class for all training examples (K by m)


a2 = sigmoid(Theta1*X');
m2 = size(a2, 2);
a2 = [ones(1, m2); a2];
a3 = sigmoid(Theta2*a2);

[M, p] = max(a3', [], 2);



% =========================================================================


end
