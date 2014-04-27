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

% build y
build_y = zeros(m, max(y));
for i = 1:m
    build_y(i, y(i,1)) = 1;
end

% forward prop
tmp = [ones(m, 1) X];
z2 = (tmp * Theta1');
a2 = sigmoid(z2);
tmp2 = [ones(m, 1) a2];
z3 = (tmp2 * Theta2');
a3 = sigmoid(z3);
[~, h] = max(a3, [], 2);

% compute the cost
for i = 1:m
    cost = (sum((build_y(i, :) .* log(a3(i, :)) + ...
        (1 - build_y(i, :)) .* log(1 - a3(i, :)))));
    J = J + cost;
end

J = (-(1 / m)) * J;

% regularization
thetas = 0;
hidden_thetas = 0;
reg = lambda / (2 * m);

tmp_theta2 = Theta2(:, 2:size(Theta2, 2));
for i = 1:size(Theta2, 1)
    thetas = thetas + sum(tmp_theta2(i, :) .^ 2);
end

tmp_theta1 = Theta1(:, 2:size(Theta1, 2));
for i = 1:size(Theta1, 1)
    hidden_thetas = hidden_thetas + sum(tmp_theta1(i, :) .^ 2);
end

J = J + reg * (thetas + hidden_thetas);

% back prop - assume 3 layers, 1 input, 1 hidden, 1 output

% iterate through each training example
for t = 1:m
    % forward

    a1 = X(t, :);

    a1 = [1 a1]; % adding bias unit
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);

    a2 = [1 a2]; % adding bias unit
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    % output layer error
    delta3 = (a3 - build_y(t, :));

    % hidden layer error
    delta2 = (delta3 * Theta2) .* [1 sigmoidGradient(z2)];
    delta2 = delta2(2:end);

    % calculate gradient
    Theta2_grad = (Theta2_grad + (delta3' * a2));
    Theta1_grad = (Theta1_grad + (delta2' * a1));
end

Theta1_grad = (1 / m) .* Theta1_grad;
Theta2_grad = (1 / m) .* Theta2_grad;

% gradient regularization - not for bias unit

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end