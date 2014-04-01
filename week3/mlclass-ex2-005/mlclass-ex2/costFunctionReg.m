function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypo = sigmoid(X * theta); % hypothesis of all training data

% Here do not regularize theta(0) which corresponds to theta(1) in Octave
J = (1 / m) * (-y' * log(hypo) - (1 - y') * log(1 - hypo)) + ...
    (lambda / (2 * m)) * (theta' * theta - theta(1)^2);

% Use this mask to make sure theta(0) is NOT regularized
mask = ones(size(theta));

% this will set theta(0) to 0, so grad(0) will have lambda/m*theta(0) = 0
mask(1) = 0; 

grad = (1 / m) * X' * (hypo - y) + (lambda / m) * (theta .* mask);

% =============================================================

end
