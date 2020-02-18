%% load data %%
clear all;
close all;

%% simmulation_1, BRISK, FV encoding 20 clusters
%Load annotations: Y
fprintf('Loading annotations...\n');
load('simulation_1_fixed.mp4_BRISK.csv_st6_ANN.mat');

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

%Load FVs: X
fprintf('Loading Fisher vectors...\n');
load('simulation_1_fixed.mp4_BRISK.csv_st6_FVs.mat');

%Abnormal class for testing
X_test = X(:,ind_Abnormal);
X_test2 = X(:,ind_PersonInteractsWithPatient);

%Normal classes for training
X_train = [X(:,ind_LyingAwakeTalkingMoving), X(:,ind_SleepingOnBack), X(:,ind_SleepingOnSide)];
clearvars X;

%% dict learning...
% dictionary dimensions
n = size(X_train, 1);
m = 500; %number of atoms must be smaller than number of training signals
% number of examples
L = size(X_train, 2);
% sparsity of each example
k = 5;
Gamma = zeros(m,L);
%% run k-svd training %%
params.data = X_train;
params.Tdata = k;
params.dictsize = m;
params.iternum = 100; %number of iterations
params.memusage = 'high';
fprintf('Starting k-svd...\n');
[Dksvd, g, err] = ksvd(params,'i');
% show results %
figure; plot(err); title('K-SVD TRAINING error convergence');
xlabel('Iteration'); ylabel('RMSE');

%% test dict on testing data %%
% perform omp %
gamma = omp(Dksvd'*X_test, Dksvd'*Dksvd, k);
%ERR(D,GAMMA) = RMSE(X,D*GAMMA) = sqrt( |X-D*GAMMA|_F^2 / numel(X) )
rmse = sqrt(sum(LSE(X_test, Dksvd, gamma))/numel(X_test));
rLSE = sqrt(LSE(X_test, Dksvd, gamma));
% show results %
figure; plot(rLSE); title('rLSE on test set: X_test');
xlabel('Sample No'); ylabel('rLSE');
hold on
mmean = movmean(rLSE, 10);
plot(mmean);
hold off

%% test dict on testing data 2 %%
% perform omp %
gamma = omp(Dksvd'*X_test2, Dksvd'*Dksvd, k);
%ERR(D,GAMMA) = RMSE(X,D*GAMMA) = sqrt( |X-D*GAMMA|_F^2 / numel(X) )
%rmse = sqrt(sum(LSE(X_test, Dksvd, gamma))/numel(X_test));
rLSE = sqrt(LSE(X_test2, Dksvd, gamma));
% show results %
figure; plot(rLSE); title('rLSE on test set: X_test2');
xlabel('Sample No'); ylabel('rLSE');
hold on
mmean = movmean(rLSE, 10);
plot(mmean);
hold off

%% test dict on training data %%
% perform omp %
%D = Dksvd;
gamma = omp(Dksvd'*X_train, Dksvd'*Dksvd, k);
%ERR(D,GAMMA) = RMSE(X,D*GAMMA) = sqrt( |X-D*GAMMA|_F^2 / numel(X) )
rmse = sqrt(sum(LSE(X_train, Dksvd, gamma))/numel(X_train));
rLSE = sqrt(LSE(X_train, Dksvd, gamma));

%% show results %%
figure; plot(rLSE); title('rLSE on train set: X');
xlabel('Sample No'); ylabel('rLSE');
hold on
mmean = movmean(rLSE, 10);
plot(mmean);
% lXb = size(Xb,2);
% yMax = max(rLSE);
% yMin = min(rLSE);
% line([lXb lXb], [yMax yMin],'Color','green','LineStyle','--')
hold off

% %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %            misc functions            %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % calculates sum((X-D*gamma).^2) in blocks
% % L2-norm loss function is also known as 
% % least squares error (LSE). It is implemented by 
% % minimizing the sum of the square of the differences 
% function result = LSE(X, D, Gamma)
%     % compute in blocks to conserve memory
%     result = zeros(1,size(X,2));
%     blocksize = 2000;
%     for i = 1:blocksize:size(X,2)
%         blockids = i : min(i+blocksize-1,size(X,2));
%         result(blockids) = sum((X(:,blockids) - D*Gamma(:,blockids)).^2);
%     end
% end

