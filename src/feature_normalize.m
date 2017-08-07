%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


function [X_norm, mu, sigma] = feature_normalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

n = size(X,2);

for ii = 1:n
  mu(ii) = mean(X(:, ii));
  sigma(ii) = std(X(:, ii));
end

X_norm = (X - mu) ./ sigma;


end
