% A learning curve plots training and
% cross validation error as a function of training set size.
% This function returns the training and cross validation
% error as m x 1 vectors  for training set sizes of 1 ... m.
% The training uses regularization parameter lambda.
function [error_train, error_val m] = ...
    learning_curve(X_train, y_train, X_cv, y_cv, lambda)

m_train = size(X_train, 1);
m_cv = size(X_cv, 1);

m = min(m_train, m_cv);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

alpha = 0.1;
num_iters = 400;

init_theta = [1 ; zeros(size(X_train, 2)-1, 1)];

     for i = 1:m
         % Compute train/cross validation errors using training examples 
         % X(1:i, :) and y(1:i), storing the result in 
         % error_train(i) and error_val(i)
%           theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
%           [J grad] = linearRegCostFunction(Xval, yval, theta, 0);
%           [J grad] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
         
%         [theta J_hist] = gradient_descent(X_train(1:i, :), y_train(1:i), ...
%         init_theta, alpha, num_iters, lambda);
         theta = train_linear_reg(X_train(1:i, :), y_train(1:i), lambda);

         
         J = cost_func(X_train(1:i, :), y_train(1:i), theta, 0);
         
         error_train(i) = J;

         J = cost_func(X_cv, y_cv, theta, 0);           
         error_val(i) = J;
         
     end
end
