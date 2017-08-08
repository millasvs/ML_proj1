%% Using linear regression on the "Computer Hardware Data Set"  from UCI:
%% http://archive.ics.uci.edu/ml/datasets/Computer+Hardware 

% there might be a more efficient way to do this, 
% but it was the first one that worked for me

clear ; close all; clc


[a, b, c, d, e, f, g, h, i, j] = textread ('machine.txt', ...
"%s %s %f %f %f %f %f %f %f %f", "delimiter", ",");

figure;
plot(f, i, '*r');
%plot(1:numel(J_hist), J_hist, '-b', 'LineWidth', 2);
xlabel('Machine cycle time');
ylabel('Published relative performance');

% 1st degree polynomial = Test error: 446.598413
% log(c) d e f g h = Test error: 435.939121
% 


% 3. MYCT: machine cycle time in nanoseconds (integer) -- log?
% 4. MMIN: minimum main memory in kilobytes (integer) -- linear?
% 5. MMAX: maximum main memory in kilobytes (integer) -- linear?
% 6. CACH: cache memory in kilobytes (integer) -- linear??
% 7. CHMIN: minimum channels in units (integer) -- linear??
% 8. CHMAX: maximum channels in units (integer) -- ??
% 9. PRP: published relative performance (integer)
%10. ERP: estimated relative performance from the original article (integer)

pause;

%X = [log(c) d e f g h];
X = f;
y = i;

% starting with zero for all features and
% an intercept term
initial_theta = [1 ; zeros(size(X, 2), 1)];

% normalize X
[X mu sigma] = feature_normalize(X);

% adding intercept term to X
X = [ones(size(X, 1), 1) X.^4];

% shuffle the rows of X
%X = X(randperm(size(X,1)),:);

% split into training, cross validation
% and test set
%[X_train y_train X_cv y_cv X_test y_test] = split_training_set(X, y);


%% fit different hypotheses
% first degree polynomial
alpha = 0.02;
num_iters = 800;
[theta J_hist] = gradient_descent(X, y, ...
initial_theta, alpha, num_iters, 0);

% second degree polynomial

% 4th degree polynomial

% 8th degree polynomial



% check test set error
predictions = X * theta;
error_test = sum((predictions-y).^2);
error_test = error_test/(2*size(X, 1));
fprintf('Test error: %f\n', error_test);




% plot the value of J_hist to see how the cost is decreasing
% to make sure gradient descent is working
%figure;
%plot(1:numel(J_hist), J_hist, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost');
%

if size(X, 2) <= 2
  % plot training data
%  plot(X,y, '*r');
  
%  figure;
%  plot(X, y, 'rx', 'MarkerSize', 10);
%  ylabel('Published relative performance'); % Set the y?axis label xlabel('Population of City in 10,000s'); % Set the x?axis label 
%  xlabel('machine cycle time in ns'); % Set the x?axis label 
%  pause;

  % plot linear fit
  hold on;
  plot(X(:,2), X*theta, '-');
  legend('Training data', 'Linear regression')
  hold off; % don't overlay any more plots on this figure
endif

%figure;
%T = rand(5,1);
%plot(T, log(T), '-');
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


