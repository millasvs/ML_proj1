function [theta] = train_linear_reg(X, y, lambda)
%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
%regularization parameter lambda
%   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.
%

% Initialize Theta
%initial_theta = zeros(size(X, 2), 1); 
init_theta = [1 ; zeros(size(X, 2)-1, 1)];

alpha = 0.1;
num_iters = 400;
% Minimize using gradient_descent
theta = gradient_descent(X, y, init_theta, alpha, num_iters, lambda);

end
