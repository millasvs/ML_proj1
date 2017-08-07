function [error_train, error_val] = ...
    learning_curve(X_train, y_train, X_cv, y_cv, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m_train = size(X_train, 1);
m_cv = size(X_cv, 1);

m = min(m_train, m_cv);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

alpha = 0.1;
num_iters = 400;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%

init_theta = [1 ; zeros(size(X_train, 2)-1, 1)];

       for i = 1:m
           % Compute train/cross validation errors using training examples 
           % X(1:i, :) and y(1:i), storing the result in 
           % error_train(i) and error_val(i)

%           theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
           
           [theta J_hist] = gradient_descent(X_train(1:i, :), y_train(1:i), ...
           init_theta, alpha, num_iters, lambda);
           J = cost_func(X_train(1:i, :), y_train(1:i), theta, lambda);
           
%           [J grad] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
           
           error_train(i) = J;

%           [J grad] = linearRegCostFunction(Xval, yval, theta, 0);
%           [theta J_hist] = gradient_descent(X_cv(1:i, :), y_cv(1:i), ...
%           init_theta, alpha, num_iters, lambda);

           J = cost_func(X_cv(1:i, :), y_cv(1:i), theta, lambda);           
           error_val(i) = J;
           
       end
end
