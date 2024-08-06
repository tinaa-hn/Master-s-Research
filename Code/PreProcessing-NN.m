
%% Concatenating the .mat files
% All = []; % Initialize an empty array to hold all structs
% for i = 1:30
%     % Generate the variable name for the current results struct
%     varName = sprintf('results%d', i);
%     
%     % Concatenate the current results struct to the All array
%     All = [All, evalin('base', varName)];
% end


%% Preprocessing data

load('Simul-Results-AfterTrain.mat', 'results2');
load('Simul2-Results.mat', 'results');

newResults = [results2, results]; 

newResults = [newResults, All];

%% Cropping data

[cropped_BinCounts, common_boundary] = crop_to_common_boundary(All);

%% Flatten the cropped_BinCounts to get input X

% Number of data

numData = size(cropped_BinCounts, 2);

input = zeros(numData, 37*24);

for i = 1:numData
    input(i, :) = reshape(cropped_BinCounts{i}, 1, []);
end


%% Construction of Output(target Y)

EA = 1; %e-6;
output = zeros(numData, 2); % 2 columns for A and B

for i = 1:numData
    output(i, 1) = All(i).A * EA; % Column for A
    output(i, 2) = All(i).B * EA; % Column for B
end

%% Normalize input and output

X_mean = mean(input);
X_std = std(input);
X_standardized = input;
nonZeroStdCols = X_std ~= 0;
X_standardized(:, nonZeroStdCols) = (input(:, nonZeroStdCols) - X_mean(nonZeroStdCols)) ./ X_std(nonZeroStdCols);


% Normalizing output Y (not necessary)
Y = output;
Y_mean = mean(Y);
Y_std = std(Y);
Y_norm = (Y - Y_mean) ./ Y_std;

%% Randomly splitting to get train and test data separated


% First we shuffle the data to break the order

X = X_standardized;
Y = Y_norm;
numDataPoints = size(X, 1); 
indices = randperm(numDataPoints);
XShuffled = X(indices, :);
YShuffled = Y(indices, :);

trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;

numTrain = floor(trainRatio * numDataPoints);
numVal = floor(valRatio * numDataPoints);

% Training set
XTrain = XShuffled(1:numTrain, :);
YTrain = YShuffled(1:numTrain, :);

% Validation set
XVal = XShuffled(numTrain+1:numTrain+numVal, :);
YVal = YShuffled(numTrain+1:numTrain+numVal, :);

% Test set
XTest = XShuffled(numTrain+numVal+1:end, :);
YTest = YShuffled(numTrain+numVal+1:end, :);

%% Cross validation and number of hidden layer neurons search

neuronsToTry = 5:20; 
cvMSE = zeros(length(neuronsToTry), 1); % To store the cross-validation mean squared error

k = 10; % Number of folds for cross-validation
indices = simple_kfold(YTrain, k); 

for n = 1:length(neuronsToTry)
    mseForEachFold = zeros(k, 1);
    
    for fold = 1:k
        testIdx = (indices == fold);
        trainIdx = ~testIdx;
        
        % NN architecture
        layers = [
            featureInputLayer(size(XTrain,2))
            fullyConnectedLayer(neuronsToTry(n))
            reluLayer
            fullyConnectedLayer(size(YTrain,2)) 
            regressionLayer
        ];
        
        % Training options 
        options = trainingOptions('adam', ...
            'MaxEpochs', 100, ...
            'ValidationFrequency', 10, ...
            'Verbose', false);
        
        % Training the network
        net = trainNetwork(XTrain(trainIdx,:), YTrain(trainIdx,:), layers, options);
        
        % Testing the network
        YPredicted = predict(net, XTrain(testIdx,:));
        mseForEachFold(fold) = mean(mean((YPredicted - YTrain(testIdx,:)).^2));
    end
    
    % Computing the mean MSE over all folds for the current number of neurons
    cvMSE(n) = mean(mseForEachFold);
    display(['Neurons tried: ', num2str(neuronsToTry(n))]);
end

% Finding the number of neurons that resulted in the lowest MSE
[~, optimalNeuronIdx] = min(cvMSE);
optimalNeurons = neuronsToTry(optimalNeuronIdx);
display(['Optimal number of neurons: ', num2str(optimalNeurons)]);

%%
figure;
plot(neuronsToTry, cvMSE, 'b-o', 'LineWidth', 2);
xlabel('Number of Neurons in Hidden Layer');
ylabel('Cross-Validation MSE');
title('Cross-Validation MSE vs. Number of Neurons');
grid on;

for i = 1:length(neuronsToTry)
    text(neuronsToTry(i), cvMSE(i), sprintf('(%d, %.4f)', neuronsToTry(i), cvMSE(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end

%% Training the Network
%optimalNeurons = 10;
% Neural Network Architecture with Optimal Number of Neurons
layers = [
    featureInputLayer(size(XTrain, 2), 'Name', 'input')
    fullyConnectedLayer(optimalNeurons, 'Name', 'fc1', 'WeightL2Factor', 0.01) % Corrected L2 regularization for weights
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(optimalNeurons, 'Name', 'fc2', 'WeightL2Factor', 0.01) % Corrected L2 regularization for weights
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2, 'Name', 'output', 'WeightL2Factor', 0.01)
    regressionLayer('Name', 'regressionOutput')
    ];


options = trainingOptions('sgdm', ...
    'MaxEpochs', 10000, ...
    'MiniBatchSize', 20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', {XVal, YVal}, ... 
    'ValidationFrequency', 200, ...
    'ValidationPatience', 5, ... 
    'LearnRateSchedule', 'piecewise', ... 
    'LearnRateDropFactor', 0.5, ... 
    'LearnRateDropPeriod', 500);




[trainedNet, trainInfo] = trainNetwork(XTrain, YTrain, layers, options);


%% Cross validation training

k = 5; 
foldIndices = simple_kfold(YTrain, k); 

valPerformances = zeros(k, 1); 

for i = 1:k
    fprintf('Training on fold %d...\n', i);

    trainIdx = foldIndices ~= i; % Training data: data not in the i-th fold
    valIdx = foldIndices == i;   % Validation data: data in the i-th fold

    XTrainFold = XTrain(trainIdx, :);
    YTrainFold = YTrain(trainIdx, :);
    XValFold = XTrain(valIdx, :);
    YValFold = YTrain(valIdx, :);

    % Updating the options to use the current fold's validation set
    options.ValidationData = {XValFold, YValFold};

    % Training the network using the current fold's training set
    [trainedNet, trainInfo] = trainNetwork(XTrainFold, YTrainFold, layers, options);

    % Storingg the validation performance
    valPerformances(i) = trainInfo.FinalValidationLoss;
end

% the average validation performance across all folds
meanValPerformance = mean(valPerformances);
fprintf('Average Validation Performance: %f\n', meanValPerformance);


%% Testing 
YPred = predict(trainedNet, XTest);

% MAE for the test set
testMAE = mean(abs(YPred - YTest));
fprintf('Test MAE: %.7e\n', testMAE);

% Predict responses for the training set
YPredTrain = predict(trainedNet, XTrain);
trainMAE = mean(abs(YPredTrain - YTrain));
fprintf('Train MAE: %.7e\n', trainMAE);

% R-squared
residuals = YTest - YPred;
SStot = sum((YTest - mean(YTest)).^2);
SSres = sum(residuals.^2);
rsquared = 1 - SSres / SStot;
fprintf('Test R-squared: %.4f\n', rsquared);

% Plot of predicted vs. actual values 
figure;
scatter(YTest, YPred, 'filled');
hold on; 
plot(YTest, YTest, 'r-', 'LineWidth', 2); 
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
title('Predicted vs. Actual Values');
grid on;
axis equal;


%% Functions

function [cropped_matrices, common_boundary] = crop_to_common_boundary(results)
    min_row = 500;
    max_row = 0;
    min_col = 500;
    max_col = 0;
    for i = 1:numel(results)
        currentBinCounts = results(i).BinCounts;
        currentBinCounts(currentBinCounts <= 2) = 0;
        results(i).BinCounts = currentBinCounts;
        
        [rows, cols] = find(currentBinCounts);
        if ~isempty(rows)
            min_row = min(min_row, min(rows));
            max_row = max(max_row, max(rows));
            min_col = min(min_col, min(cols));
            max_col = max(max_col, max(cols));
        end
    end

    % The common boundary is determined by the maximum extent of non-zero values
    common_boundary = [min_row, max_row, min_col, max_col];

    %cropping all matrices to this common boundary
    cropped_matrices = cell(1, numel(results));
    for i = 1:numel(results)
        currentBinCounts = results(i).BinCounts;
        cropped_matrices{i} = currentBinCounts(min_row:max_row, min_col:max_col);
    end
end

function indices = simple_kfold(Y, k)
    n = size(Y, 1); 
    indices = zeros(n, 1); 
    randIndices = randperm(n); 

    for i = 1:k
        indices(randIndices(((floor((i-1)*n/k) + 1):floor(i*n/k)))) = i;
    end
end
