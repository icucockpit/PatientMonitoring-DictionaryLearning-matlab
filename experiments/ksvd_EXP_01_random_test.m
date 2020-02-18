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
fprintf('Loading data...\n');

% sparsity of each example
load('X_test_Abnormal.mat'); % -> X_test_Abnormal
load('X_test_Abnormal_FRAMES.mat'); % -> X_test_Abnormal_FRAMES
load('X_test_PersonInteractsWithPatient.mat'); % -> X_test_PersonInteractsWithPatient
load('X_test_PersonInteractsWithPatient_FRAMES.mat'); % -> X_test_PersonInteractsWithPatient_FRAMES
load('X_test_SubsampledFromTrainingSet.mat'); % -> X_test_SubsampledFromTrainingSet
load('X_test_SubsampledFromTrainingSet_FRAMES.mat'); % -> X_test_SubsampledFromTrainingSet_FRAMES
load('X_train_Subsampled.mat'); % -> X_train_Subsampled
load('X_train_Subsampled_FRAMES.mat'); % -> X_train_Subsampled_FRAMES
load('DICTIONARY.mat'); % -> Dksvd
%load('DICTIONARY_ERROR.mat'); % -> err
load('ksvd_params.mat'); % -> noAtoms, sparsity, noIterations

%find max framenumber for plots
max_framenumber = 0;
mxtmp = max(X_test_Abnormal_FRAMES(:,2));
if ( mxtmp >= max_framenumber)
    max_framenumber = mxtmp;
end
mxtmp = max(X_test_PersonInteractsWithPatient_FRAMES(:,2));
if ( mxtmp >= max_framenumber)
    max_framenumber = mxtmp;
end
mxtmp = max(X_test_SubsampledFromTrainingSet_FRAMES(:,2));
if ( mxtmp >= max_framenumber)
    max_framenumber = mxtmp;
end
mxtmp = max(X_train_Subsampled_FRAMES(:,2));
if ( mxtmp >= max_framenumber)
    max_framenumber = mxtmp;
end

% show results %
%figure; plot(err); title('K-SVD training error convergence');
%xlabel('Iteration'); ylabel('RMSE');

%% test dict on X_train_Subsampled ACTUAL TRAINING SET %%
% perform omp %
gamma = omp(Dksvd'*X_train_Subsampled, Dksvd'*Dksvd, sparsity);
%rmse = sqrt(sum(LSE(X_train_Subsampled, Dksvd, gamma))/numel(X_train));
rLSE = sqrt(LSE(X_train_Subsampled, Dksvd, gamma));
figure;
plot(median(X_train_Subsampled_FRAMES,2), rLSE, '.');
title('rLSE on X\_train\_Subsampled ACTUAL TRAINING SET');
xlabel('Frame No'); ylabel('rLSE');
ylim([0 4500]);
xlim([0 max_framenumber]);
% hold on
% mmean = movmean(rLSE, 10);
% plot(mmean);
% hold off

%% test dict on X_test_SubsampledFromTrainingSet %%
% perform omp %
gamma = omp(Dksvd'*X_test_SubsampledFromTrainingSet, Dksvd'*Dksvd, sparsity);
%rmse = sqrt(sum(LSE(X_test, Dksvd, gamma))/numel(X_test));
rLSE = sqrt(LSE(X_test_SubsampledFromTrainingSet, Dksvd, gamma));
% show results %
figure;
plot(median(X_test_SubsampledFromTrainingSet_FRAMES,2), rLSE, '.');
title('rLSE on X\_test\_SubsampledFromTrainingSet');
xlabel('Frame No'); ylabel('rLSE');
ylim([0 4500]);
xlim([0 max_framenumber]);
% hold on
% mmean = movmean(rLSE, 10);
% plot(mmean);
% hold off

%% test dict on X_test_Abnormal %%
% perform omp %
gamma = omp(Dksvd'*X_test_Abnormal, Dksvd'*Dksvd, sparsity);
%rmse = sqrt(sum(LSE(X_test_Abnormal, Dksvd, gamma))/numel(X_test));
rLSE = sqrt(LSE(X_test_Abnormal, Dksvd, gamma));
% show results %
figure;
plot(median(X_test_Abnormal_FRAMES,2), rLSE, '.');
title('rLSE on X\_test\_Abnormal');
xlabel('Frame No'); ylabel('rLSE');
ylim([0 4500]);
xlim([0 max_framenumber]);
% hold on
% mmean = movmean(rLSE, 10);
% plot(mmean);
% hold off

%% test dict on X_test_PersonInteractsWithPatient %%
% perform omp %
gamma = omp(Dksvd'*X_test_PersonInteractsWithPatient, Dksvd'*Dksvd, sparsity);
%rmse = sqrt(sum(LSE(X_test, Dksvd, gamma))/numel(X_test));
rLSE = sqrt(LSE(X_test_PersonInteractsWithPatient, Dksvd, gamma));
% show results %
figure;
plot(median(X_test_PersonInteractsWithPatient_FRAMES,2), rLSE, '.');
title('rLSE on X\_test\_PersonInteractsWithPatient');
xlabel('Frame No'); ylabel('rLSE');
ylim([0 4500]);
xlim([0 max_framenumber]);
% hold on
% mmean = movmean(rLSE, 10);
% plot(mmean);
% hold off


