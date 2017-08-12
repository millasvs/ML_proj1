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

%init_theta = [1 ; zeros(size(X_train, 2)-1, 1)];

     for i = 1:m
      for j = 1:50
         % Compute train/cross validation errors using training examples 
         % X(1:i, :) and y(1:i), storing the result in 
         % error_train(i) and error_val(i)
%           theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
%           [J grad] = linearRegCostFunction(Xval, yval, theta, 0);
%           [J grad] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
         
%         [theta J_hist] = gradient_descent(X_train(1:i, :), y_train(1:i), ...
%         init_theta, alpha, num_iters, lambda);
      
   
%Concretely, to determine the training error and cross validation error for
%i examples, you should first randomly select i examples from the training set
%and i examples from the cross validation set. You will then learn the param-
%eters θ using the randomly chosen training set and evaluate the parameters
%θ on the randomly chosen training set and cross validation set. The above
%steps should then be repeated multiple times (say 50) and the averaged error
%should be used to determine the training error and cross validation error for
%i examples.
%For this optional (ungraded) exercise, you should implement the above
%strategy for computing the learning curves. For reference, figure 10 shows the
%learning curve we obtained for polynomial regression with λ = 0.01. Your
%figure may differ slightly due to the random selection of examples.
%You do not need to submit any solutions for this optional (ungraded)
%exercise.

         [rand_X_train rand_y_train] = rand_select_i(X_train, y_train, i);
         

         [rand_X_cv rand_y_cv] = rand_select_i(X_cv, y_cv, i);
         
         

         theta = train_linear_reg(rand_X_train, rand_y_train, lambda);

         
         J = cost_func(rand_X_train, rand_y_train, theta, 0);
         
         error_train(i) = J;

         J = cost_func(X_cv, y_cv, theta, 0);           
         error_val(i) = J;
         
         
      end 
         
         
     end
end
