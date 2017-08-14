% Batch gradient descent - runs gradient descent over all training
% examples. Runs for num_iter iterations with learning rate alpha
% and regularization parameter lambda.

% Stores the intermediate values of the cost function in J_hist, 
% as a means to test gradient descent with different learning rates
% and verify that it's working.

function [theta J_hist] = gradient_descent(X, y, init_theta, alpha, num_iters, lambda)
    theta = init_theta;
    J_hist = zeros(num_iters, 1);
    n = size(X, 2); % number of features
    m = size(X, 1); % number of training examples
    
    for iter = 1:num_iters

      % temp: stores the value of the gradient
      temp = theta;
      for ii = 1:n
        diff = X * theta - y;
        temp(ii) = (alpha/m) * sum( diff' * X(:, ii));
      end
      
      theta = theta - temp;

%TODO: introduce lambda to the regularization itself      
      reg = (lambda*theta) / m;
      reg(1) = 0;
      theta = theta + reg;

      
      % Save the cost J in every iteration    
      J_hist(iter) = cost_func(X, y, theta, lambda);

    end

end