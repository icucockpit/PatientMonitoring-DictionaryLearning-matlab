clear all;
close all;

if isunix
    % set paths
    addpath('/home/ubuntu/MATLAB/CODE/DICT_LEARNING_LIB/ompbox10') 
    addpath('/home/ubuntu/MATLAB/CODE/DICT_LEARNING_LIB/ksvdbox13')
elseif ispc
    % set paths
    addpath('C:\Users\AED\Desktop\MP_WORK\MATLAB\CODE\DICT_LEARNING_LIB\ompbox10') 
    addpath('C:\Users\AED\Desktop\MP_WORK\MATLAB\CODE\DICT_LEARNING_LIB\ksvdbox13')
else
    disp('Platform not supported')
end


%% load data %%
%% simmulation_1, BRISK, FV encoding 20 clusters
%Load annotations: Y
fprintf('Loading annotations...\n');
load('simulation_1_fixed.mp4_BRISK.csv_st6_ANN.mat'); % -> Y

rows = Y.attr_a=='EmptyBed';
ind_EmptyBed = find(rows == 1);
rows = Y.attr_b=='LyingAwakeTalkingMoving';
ind_LyingAwakeTalkingMoving = find(rows == 1);
rows = Y.attr_c=='SleepingOnBack';
ind_SleepingOnBack = find(rows == 1);
rows = Y.attr_d=='SleepingOnSide';
ind_SleepingOnSide = find(rows == 1);
rows = Y.attr_e=='PersonInteractsWithPatient';
ind_PersonInteractsWithPatient = find(rows == 1);
rows = Y.attr_f=='Abnormal';
ind_Abnormal = find(rows == 1);

%Load frames
load('simulation_1_fixed.mp4_BRISK.csv_st6_FRAMES.mat'); % -> frames

%Load FVs: X
fprintf('Loading Fisher vectors...\n');
load('simulation_1_fixed.mp4_BRISK.csv_st6_FVs.mat'); % -> X

%Split
%Abnormal class for testing
X_test_Abnormal = X(:,ind_Abnormal);
X_test_Abnormal_FRAMES = frames(ind_Abnormal,:);
fprintf('Saving X_test_Abnormal...\n');
save('X_test_Abnormal.mat','X_test_Abnormal','-v7.3');
save('X_test_Abnormal_FRAMES.mat','X_test_Abnormal_FRAMES','-v7.3');
X_test_PersonInteractsWithPatient = X(:,ind_PersonInteractsWithPatient);
X_test_PersonInteractsWithPatient_FRAMES = frames(ind_PersonInteractsWithPatient,:);
fprintf('Saving X_test_PersonInteractsWithPatient...\n');
save('X_test_PersonInteractsWithPatient.mat','X_test_PersonInteractsWithPatient','-v7.3');
save('X_test_PersonInteractsWithPatient_FRAMES.mat','X_test_PersonInteractsWithPatient_FRAMES','-v7.3');

%Normal classes for training
fprintf('Creating training set...\n');
X_train = [X(:,ind_LyingAwakeTalkingMoving), X(:,ind_SleepingOnBack), X(:,ind_SleepingOnSide)];
X_train_FRAMES = [frames(ind_LyingAwakeTalkingMoving,:); frames(ind_SleepingOnBack,:); frames(ind_SleepingOnSide,:)];
clearvars X;

%Split randomly
fprintf('Random split of training set...\n');
p = .7;      % proportion of rows to select for training
N = size(X_train,2);  % total number of cols 
tf = false(N, 1);    % create logical index vector
tf(1:round(p*N)) = true;    
tf = tf(randperm(N));   % randomise order
X_train_Subsampled = X_train(:,tf);
X_train_Subsampled_FRAMES = X_train_FRAMES(tf,:);
X_test_SubsampledFromTrainingSet = X_train(:,~tf);
X_test_SubsampledFromTrainingSet_FRAMES = X_train_FRAMES(~tf,:);

fprintf('Saving X_train_Subsampled...\n');
save('X_train_Subsampled.mat','X_train_Subsampled','-v7.3');
save('X_train_Subsampled_FRAMES.mat','X_train_Subsampled_FRAMES','-v7.3');
fprintf('Saving X_test_SubsampledFromTrainingSet...\n');
save('X_test_SubsampledFromTrainingSet.mat','X_test_SubsampledFromTrainingSet','-v7.3');
save('X_test_SubsampledFromTrainingSet_FRAMES.mat','X_test_SubsampledFromTrainingSet_FRAMES','-v7.3');

% clearvars X_train;
% clearvars X_train_FRAMES;

%% dict learning... k-svd
X_train = X_train_Subsampled;
noAtoms = 700; %number of atoms must be smaller than number of training signals
sparsity = 10; % sparsity of each example
noIterations = 100; %number of iterations
fprintf('Saving k-svd parameters...\n');
save('ksvd_params.mat','noAtoms','sparsity','noIterations','-v7.3');
% dictionary dimensions
%n = size(X_train, 1);
%L = size(X_train, 2); % number of examples
%Gamma = zeros(noAtoms,L); %sparce coefficients
params.data = X_train;
params.Tdata = sparsity;
params.dictsize = noAtoms;
params.iternum = noIterations;
params.memusage = 'high';
fprintf('Starting k-svd...\n');
[Dksvd, g, err] = ksvd(params,'i');

fprintf('Saving dictionary and error...\n');
save('DICTIONARY.mat','Dksvd','-v7.3');
save('DICTIONARY_ERROR.mat','err','-v7.3');
