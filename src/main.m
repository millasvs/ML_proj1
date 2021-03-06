%% Using linear regression on the "Computer Hardware Data Set"  from UCI:
%% http://archive.ics.uci.edu/ml/datasets/Computer+Hardware 

clear ; close all; clc

% there might be a more efficient way to get data from the file, 
% but it was the first one that worked for me
[a, b, c, d, e, f, g, h, i, j] = textread ('machine.txt', ...
"%s %s %f %f %f %f %f %f %f %f", "delimiter", ",");

% 3. MYCT: machine cycle time in nanoseconds (integer) -- log?
% 4. MMIN: minimum main memory in kilobytes (integer) -- linear?
% 5. MMAX: maximum main memory in kilobytes (integer) -- linear?
% 6. CACH: cache memory in kilobytes (integer) -- linear??
% 7. CHMIN: minimum channels in units (integer) -- linear??
% 8. CHMAX: maximum channels in units (integer) -- ??
% 9. PRP: published relative performance (integer)
%10. ERP: estimated relative performance from the original article (integer)


%% prepare data set
  
  h = poly_features(h, 2);
  Q = mapFeature(d, e);
  Q = Q(:,2:end);
  X = [c Q f g h];
  y = i;
  
  
  % normalize X
  [X mu sigma] = feature_normalize(X);
  
  % adding intercept term to X
  X = [ones(size(X, 1), 1) X];
  
  %% split into training, cross validation
  % and test set (shuffles rows first)
  [X_train y_train X_cv y_cv X_test y_test] = split_training_set(X, y);


%% gradient descent
  % starting with zero for all features and
  % an intercept term
  n = size(X, 2); % number of features  
  init_theta = [1 ; zeros(n-1, 1)];
  alpha = 0.01;
  num_iters = 150;
  
  [theta J_hist] = gradient_descent(X_train, y_train, ...
  init_theta, alpha, num_iters, 0);
  %% plot the value of J_hist to see how the cost is decreasing
  % to make sure gradient descent is working
  figure;
  plot(1:numel(J_hist), J_hist, '-b', 'LineWidth', 2);
  xlabel('Number of iterations');
  ylabel('Cost');
  hold off;
  
  fprintf('Program paused. Press enter to continue.\n');
  pause;

%% check cross validation set error
  predictions = X_cv * theta;
  error_test = sum((predictions-y_cv).^2);
  error_test = error_test/(2*size(X_cv, 1));  
  fprintf('Cross validation error: %f\n', error_test);

%% check test set error
  predictions = X_test * theta;
  error_test = sum((predictions-y_test).^2);
  error_test = error_test/(2*size(X_test, 1));
  fprintf('Test set error: %f\n', error_test);

%% if possible - plot linear regression fit -
% works if there is only one feature
if size(X, 2) <= 2
  figure;
  plot(X(:,2), y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
  xlabel('Machine cycle time in ns');
  ylabel('Published relative performance');
  hold on;
  plot(X(:,2), X*theta, '--', 'LineWidth', 2);
  legend('Training data', 'Linear regression')
  hold off;
endif


%%=========== Plot learning curves ===========%%

% Plot cross validation error
  
  m = size(X_train, 1);
  lambda = 9;
  [error_train, error_val] = ...
      learning_curve(X_train, y_train, ...
                    X_cv, y_cv, ...
                    lambda);
                    
  figure;                  
  plot(1:m, error_train, 1:m, error_val);
  title('Learning curve for linear regression')
  legend('Train', 'Cross Validation')
  xlabel('Number of training examples')
  ylabel('Error')
  axis([0 126 0 10000])

  fprintf('Program paused. Press enter to continue.\n');
  pause;

  fprintf('# Training Examples  \tTrain Error \tCross Validation Error\n');
  for i = 1:m
      fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
  end

  fprintf('Program paused. Press enter to continue.\n');
  pause;

% Plot test set error
  
  [error_train, error_test] = ...
      learning_curve(X_train, y_train, ...
                    X_test, y_test, ...
                    lambda);
                    
  figure;                  
  plot(1:m, error_train, 1:m, error_test);
  title('Learning curve for linear regression')
  legend('Train', 'Test set')
  xlabel('Number of training examples')
  ylabel('Error')
  axis([0 126 0 10000])

  fprintf('Program paused. Press enter to continue.\n');
  pause;

  fprintf('# Training Examples  \tTrain Error \tTest set Error\n');
  for i = 1:m
      fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_test(i));
  end

  fprintf('Program paused. Press enter to continue.\n');
  pause;
  
  
%%======== Selecting lambda using cross validation set ======%%  


%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validation_curve(X_train, y_train, X_cv, y_cv);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t \tTrain Error \tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f  \t  %f\t  %f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

[M MI] = min(error_val);

best_lambda = lambda_vec(MI);

fprintf('Best lambda value found through validationCurve = %f.\n', best_lambda);
           
theta = gradient_descent(X_train, y_train, init_theta, ...
alpha, num_iters, best_lambda);

[J grad] = cost_func(X_test, y_test, theta, 0)

fprintf('Test error for best lambda = %f.\n', J);

