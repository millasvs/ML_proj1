%% Using linear regression on the "Computer Hardware Data Set"  from UCI:
%% http://archive.ics.uci.edu/ml/datasets/Computer+Hardware 

% there might be a more efficient way to do this, 
% but it was the first one that worked for me

clear ; close all; clc


[a, b, c, d, e, f, g, h, i, j] = textread ('machine.txt', ...
"%s %s %f %f %f %f %f %f %f %f", "delimiter", ",");

X = [c d e f g h];
y = i;

% starting with zero for all features and
% an intercept term
initial_theta = [1 ; zeros(size(X, 2), 1)];

% normalize X
[X mu sigma] = feature_normalize(X);

% adding intercept term to X
X = [ones(size(X, 1), 1) X];

% shuffle the rows of X
%X = X(randperm(size(X,1)),:);

% divide into training, cross-validation,
% test set and results
% training set: 60%
% cross validation set: 20%
% test set: 20%
m = size(X, 1); % number of training examples
mtrain = ceil(0.6*m);
mcv = floor((m-mtrain)/2) + mtrain + 1;

X_train = X(1:mtrain, :);
X_cv = X(mtrain+1:mcv, :);
X_test = X(mcv+1:m, :);

y_train = y(1:mtrain, :);
y_cv = y(mtrain+1:mcv, :);
y_test = y(mcv+1:m, :);


%% fit different hypotheses
% first degree polynomial
alpha = 0.1;
num_iters = 400;
[theta J_hist] = gradient_descent(X_train, y_train, ...
initial_theta, alpha, num_iters, 0);

% second degree polynomial

% 4th degree polynomial

% 8th degree polynomial



% check test set error
predictions = X_cv * theta;
error_test = sum((predictions-y_cv).^2);
error_test = error_test/(2*mcv);
fprintf('Test error: %f\n', error_test);




% plot the value of J_hist to see how the cost is decreasing
% to make sure gradient descent is working
%figure;
%plot(1:numel(J_hist), J_hist, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost');
%
%% make a prediction 
%
%
%% plot learning curves to debug learning algorithm
%lambda = 0;
%[error_train, error_val] = ...
%    learning_curve(X_train, y_train, ...
%                  X_cv, y_cv, ...
%                  lambda);

%plot(1:size(error_train,1), error_train, 1:size(error_val,1), error_val);
%title('Learning curve for linear regression')
%legend('Train', 'Cross Validation')
%xlabel('Number of training examples')
%ylabel('Error')
%axis([0 13 0 150])


