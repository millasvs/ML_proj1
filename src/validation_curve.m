%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
function [lambda_vec, error_train, error_val] = ...
    validation_curve(X_train, y_train, X_cv, y_cv)

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

init_theta = [1 ; zeros(size(X_train, 2)-1, 1)];  
alpha = 0.01;
num_iters = 150;

  for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);

    % train using gradient descent
%    theta = gradient_descent(X_train, y_train, ...
%    init_theta, alpha, num_iters, lambda);

    theta = train_linear_reg(X_train, y_train, lambda);
    J = cost_func(X_train, y_train, theta, 0);
    error_train(i) = J;

    J = cost_func(X_cv, y_cv, theta, 0);
    error_val(i) = J;



  end


end
