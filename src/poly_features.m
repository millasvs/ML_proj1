%% Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%
function [X_poly] = poly_features(X, p)

rectly.
X_poly = zeros(numel(X), p);

m = size(X, 1);

for row = 1:m
  for col = 1:p
    X_poly(row, col) = X(row).^col;
  end
end



end
