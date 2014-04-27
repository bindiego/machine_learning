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

% cost function
hypo = X*theta; % hypothesis of all training data


% Here do not regularize theta(0) which corresponds to theta(1) in Octave
J = 1/(2*m) * ((hypo - y)' * (hypo - y)) + ...
    (lambda / (2 * m)) * (theta' * theta - theta(1)^2);

% Use this mask to make sure theta(0) is NOT regularized
mask = ones(size(theta));

% this will set theta(0) to 0, so grad(0) will have lambda/m*theta(0) = 0
mask(1) = 0; 

grad = (1 / m) * X' * (hypo - y) + (lambda / m) * (theta .* mask);
% =========================================================================

grad = grad(:);

end
