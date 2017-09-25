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

s = 0;
h = sigmoid(X * theta); %computing h for every line

%looping every line and 
for i = 1:m
    s = s + ( -y(i) * log(h(i)) - (1 - y(i)) * log(1 - h(i)) );
end

%computing the regularization element
n = size(X, 2); reg_sum = 0;
for i = 2:n
    reg_sum = reg_sum + theta(i) * theta(i);
end
reg = (lambda / (2 * m)) * reg_sum;
%-----------------------------------
J = (1 / m) * s + reg;


%vectorized implementation of gradient
grad = (1 / m) * (X' * (h - y));
%-------------------------------------

%regularizing gradient
for i = 2:n
    grad(i) = grad(i) + (lambda / m) * theta(i);
end
%-----------------------------------

% =============================================================

end
