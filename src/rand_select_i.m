%% Randomly select i training examples from X and y data sets,
% where X are the features and y is the thing we want to predict

function [X_i y_i] = rand_select_i(X, y, i)

  % shuffle the rows of X and y
  all = [X y];
  all = all(randperm(size(all,1)),:);

  X = all(:, 1:size(X,2));
  y = all(:, size(X,2)+1:end);
  
  % select first i examples
  X_i = X(1:i, :);
  y_i = y(1:i, :);


end
