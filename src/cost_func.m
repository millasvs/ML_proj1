% Computes cost for regularized multivariate linear regression.
% It computes the cost for using theta as the parameter
% for fitting the data points in X and y, with regularization
% parameter lambda

function J = cost_func(X, y, theta, lambda)

%  h = [ones(size(X, 1), 1) X] * theta;
  m = size(X, 1);

  J = sum( (X * theta - y).^2 ) / (2 * m);

  reg = (lambda/(2*m)) * sum(theta.^2);

  J = J + reg;


  % alternative implementation

  %J = (1/(2*m)) * (X*theta-y)' * (X*theta-y);

end