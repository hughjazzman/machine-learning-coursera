function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % m is training sets, n is features
    % X is m by n
    % theta is n by 1
    % y is m by 1
    
    % h = X*theta (m by 1) is hypothesis vector h(x) of all training sets
    % diff = h - y to find the h(x) - y (m by 1)
    % sumvctr = diff'*X is row vector of all the sums
    % transpose diff multiply with X to get sum[(h(x)-y)x] for all features
    % (1 by m) * (m by n) = (1 by n)
    % transpose to put in theta (n by 1)
    % finally  theta - alpha/m * sum

    h = X*theta;
    diff = h - y;
    sumvctr = diff' * X;
    
    theta = theta - (alpha/m)*sumvctr';









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
