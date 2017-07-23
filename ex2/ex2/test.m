function [ output_args ] = test( input_args )
%TEST Summary of this function goes here
%   Detailed explanation goes here

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
[m, n] = size(X);
% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = ones(n + 1, 1);



costFunctionReg(initial_theta, X, y, 1000)




end

