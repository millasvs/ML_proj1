function [theta] = normal_eqn(X, y)
% analytically computes theta for linear regression
%theta = zeros(size(X, 2), 1);

theta = inv(X'*X) * X' * y;


end
