function [theta] = train_linear_reg(X, y, lambda)
%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
%regularization parameter lambda
%   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
%   the dataset (X, y) and regularization parameter lambda. Returns the
%   trained parameters theta.
%

% Initialize Theta
initial_theta = zeros(size(X, 2), 1); 

% Create "short hand" for the cost function to be minimized
costFunction = @(t) cost_func(X, y, t, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimset('num_iters', 200, 'alpha', 0.1, 'GradObj', 'on');

% Minimize using fmincg
theta = gradient_descent(costFunction, initial_theta, options);

end
