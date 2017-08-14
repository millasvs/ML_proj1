% A learning curve plots training and cross validation 
% error as a function of training set size.
% This function returns the training and cross validation
% error as m x 1 vectors  for training set sizes of 1 ... m.
% The training uses regularization parameter lambda, and
% for earch training set size i it chooses i random examples.
function [error_train, error_val] = ...
learning_curve(X_train, y_train, X_cv, y_cv, lambda)

  m_train = size(X_train, 1);

  error_train = zeros(m_train, 1);
  error_val   = zeros(m_train, 1);

  init_theta = [1 ; zeros(size(X_train, 2)-1, 1)];

  temp_error_train = zeros(1, 50);
  temp_error_val = zeros(1, 50);
  
  alpha = 0.01;
  num_iters = 150;
  for i = 1:m_train
  
  % the training and cross validation errors are given as 
  % the average value of i randomly chosen examples. a 
  % random sample is given 50 times over, and the error
  % is given as the mean of those 50 times, to minimise
  % the risk of outliers.  
    for j = 1:50

      % randomly pick i examples from training and cross validation/
      % test set
      [rand_X_train rand_y_train] = rand_select_i(X_train, y_train, i);

      % train them using gradient descent
      theta = gradient_descent(rand_X_train, rand_y_train, ...
      init_theta, alpha, num_iters, lambda);
      
      J = cost_func(rand_X_train, rand_y_train, theta, 0);
      temp_error_train(j) = J;

      J = cost_func(X_cv, y_cv, theta, 0);
      temp_error_val(j) = J;
       
    end

    error_train(i) = mean(temp_error_train);
    error_val(i) = mean(temp_error_val); 
       
  end
end
