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

n = size(X, 2);
K = size(Theta2,1);
K1 = size(Theta1,1);

% m: training sets
% n: features
% K: no. of classes (10 digits)

% Part 1 - Feed Forward

% adding bias unit to X, then transposing to get each feature in a row 
% before: X (m by n), after: X (n+1 by m)
X = [ones(m, 1) X];
X = X';

% Theta1: rows correspond to hidden unit in next layer (K1 by n + 1)
% Theta1 (K1 by n+1) * X (n+1 by m) -> a2 (K1 by m)
% bias unit added to a2 (K1 + 1 by m)
z2 = Theta1*X;
a2 = sigmoid(z2);
a2 = [ones(1,m); a2];

% Theta2: rows correspond to final output classes (K by K1+1)
% Theta2 (K by K1+1) * a2 (K1+1 by m) -> h (K by m)
% h: rows correspond to class, columns to a single training set

z3 = Theta2*a2;
h = sigmoid(z3);

% y is originally (m by 1) vector of numbers
% we want to convert to (K by m)
% recode so that each column is corresponding K dimensional vector
% with index of 1 as the digit
% have a 1 to K column vector
% compare with given y tranposed to get (K by m) logical array

Kvctr = (1:K)';
Y = (y' == Kvctr);

% Y and h are (K by m)
% mtx contains all the values to sum up, using formula for J(theta)
mtx = -Y.*log(h) - (1-Y).*log(1-h);
J = sum(mtx, 'all')/m;

% using formula - square all the values except bias

Jreg = sum(Theta1(:,2:end).^2, 'all');
Jreg = Jreg + sum(Theta2(:,2:end).^2, 'all');

Jreg = (lambda/(2*m)) * Jreg;

J = J + Jreg;

% Part 2 - Backpropogation

% h and y are both (K by m), each column corresponding to a trainig example
% D2 correspinds to gradient for Theta2 (K by K1+1)
% D1 corresponds to gradient for Theta1 (K1 by n+1)

D2 = zeros(K, K1+1);
D1 = zeros(K1, n+1);

% delta_3 = a_3 - y (K by 1)
% Theta2' (K1+1 by K) * delta_3 (K by 1) -> (K1+1 by 1)
% remove delta0_2 (K1 by 1)
% delta_3 (K by 1) * a2(:,i)' (1 by K1+1) -> D2 (K by K1+1)
% delta_2 (K1 by 1) * X(:,i)' (1 by n+1) -> D1 (K1 by n+1)

for i=1:m
    % delta_3 set
    delta_3 = (h(:,i) - Y(:, i));
    delta_2 = Theta2'*delta_3;
    
    % D2 calculated
    D2 = D2 + delta_3*a2(:,i)';
    
    % delta_2 set
    delta_2 = delta_2(2:end).*sigmoidGradient(z2(:,i));
    
    % D1 calculated
    D1 = D1 + delta_2*X(:,i)';
end

Theta1_grad = D1/m;
Theta2_grad = D2/m;


% Part 3 - Regularized Neural Networks

% add regularization to all except first column (bias unit)

Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end)];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end)];



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
