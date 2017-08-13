% analytically computes theta for linear regression
function [theta] = normal_eqn(X, y)


theta = inv(X'*X) * X' * y;



end
