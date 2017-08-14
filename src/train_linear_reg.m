%% Use built-in function fminunc to minimise cost function
%% and find optimal theta

function theta = train_linear_reg(X, y, lambda)

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

init_theta = [1 ; zeros(size(X, 2)-1, 1)];


%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(cost_func(X, y, t, lambda)), init_theta, options);

  
end  