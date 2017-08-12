% Computes cost for regularized multivariate linear regression.
% It computes the cost for using theta as the parameter
% for fitting the data points in X and y, with regularization
% parameter lambda

function [J grad] = cost_func(X, y, theta, lambda)

  % compute cost (sum of squared errors)
  m = size(X, 1);
%  J = sum( (X * theta - y).^2 ) / (2 * m);

  % alternative implementation of cost :
  J = (1/(2*m)) * (X*theta-y)' * (X*theta-y);

  % add regularization
  reg = (lambda/(2*m)) * sum(theta.^2);
  J = J + reg;
  
  %%grad
  hypothesis = X * theta;
  diff = hypothesis - y;
  grad = ( X' * diff ) / m;

% add regularization to grad
  theta(1) = 0;
  grad_reg = (lambda*theta) / m;
  grad = grad + grad_reg;
  
  

end