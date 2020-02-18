%% load data %%
clear all;
close all;

%% Dense vid1-vid2
%TRAIN
Xa = load('vid1_NORMAL_02.avi.csv_st6_FVs.csv');
Xb = load('vid2_NORMAL_01.avi.csv_st6_FVs.csv');
Xc = load('vid3_NORMAL_01.avi.csv_st6_FVs.csv');
%X = [Xa, Xb, Xc]; %concatenate
X = [Xb, Xc]; %concatenate
%clearvars Xa Xb Xc;

%TEST
Xb_test = load('vid1_ABNORMAL_01.avi.csv_st6_FVs.csv');
Xc_test = load('vid3_ABNORMAL_01.avi.csv_st6_FVs.csv');

X_test = [Xa, Xb_test, Xc_test];
%X_test = [Xb_test, Xc_test];
%clearvars Xa_test Xb_test Xc_test;

%% Dense

% Xa = load('VIRAT_S_000101_NORMAL_01.avi.csv_FVs.csv');
% Xb = load('VIRAT_S_000101_NORMAL_02.avi.csv_FVs.csv');
% %Xc = load('VIRAT_S_000101_NORMAL_03.avi.csv_FVs.csv');
% %X = [Xa, Xb, Xc]; %concatenate
% X = [Xa, Xb]; %concatenate
% %clearvars Xa Xb Xc;
% 
% Xa_test = load('VIRAT_S_000101_NORMAL_03.avi.csv_FVs.csv');
% %Xa_test = load('VIRAT_S_000101_ABNORMAL_01.avi.csv_FVs.csv');
% Xb_test = load('VIRAT_S_000101_ABNORMAL_02.avi.csv_FVs.csv');
% Xc_test = load('VIRAT_S_000101_ABNORMAL_03.avi.csv_FVs.csv');
% 
% X_test = [Xa_test, Xb_test, Xc_test];
% %clearvars Xa_test Xb_test Xc_test;

%% STIP

% Xa = load('STIP_VIRAT_S_000101_NORMAL_01.avi.csv_FVs.csv');
% Xb = load('STIP_VIRAT_S_000101_NORMAL_02.avi.csv_FVs.csv');
% Xc = load('STIP_VIRAT_S_000101_NORMAL_03.avi.csv_FVs.csv');
% X = [Xa, Xb, Xc]; %concatenate
% %X = [Xa, Xb]; %concatenate
% clearvars Xa Xb Xc;
% 
% Xa_test = load('STIP_VIRAT_S_000101_NORMAL_03.avi.csv_FVs.csv');
% %Xa_test = load('STIP_VIRAT_S_000101_ABNORMAL_01.avi.csv_FVs.csv');
% Xb_test = load('STIP_VIRAT_S_000101_ABNORMAL_02.avi.csv_FVs.csv');
% Xc_test = load('STIP_VIRAT_S_000101_ABNORMAL_03.avi.csv_FVs.csv');
% 
% X_test = [Xa_test, Xb_test, Xc_test];
% clearvars Xa_test Xb_test Xc_test;


%% dict learning...
% dictionary dimensions
n = size(X, 1);
m = 30; %number of atoms must be smaller than number of training signals

% number of examples
L = size(X, 2);

% sparsity of each example
k = 4;

Gamma = zeros(m,L);
%% run k-svd training %%

params.data = X;
params.Tdata = k;
params.dictsize = m;
params.iternum = 100; %number of iterations
params.memusage = 'high';

[Dksvd, g, err] = ksvd(params,'');

%% show results %%
figure; plot(err); title('K-SVD TRAINING error convergence');
xlabel('Iteration'); ylabel('RMSE');

%% test dict on testing data %%

% perform omp %
D = Dksvd;
gamma = omp(D'*X_test, D'*D, k);
%ERR(D,GAMMA) = RMSE(X,D*GAMMA) = sqrt( |X-D*GAMMA|_F^2 / numel(X) )
rmse = sqrt(sum(LSE(X_test, D, gamma))/numel(X_test));

%rLSE = sqrt(sum((X_test-D*gamma).^2));
rLSE = sqrt(LSE(X_test, D, gamma));

%% show results %%
figure; plot(rLSE); title('rLSE on test set: X_test');
xlabel('Sample No'); ylabel('rLSE');
hold on
mmean = movmean(rLSE, 10);
plot(mmean);
lXa = size(Xa,2);
yMax = max(rLSE);
yMin = min(rLSE);
line([lXa lXa], [yMax yMin],'Color','green','LineStyle','--')
lXb_test = lXa + size(Xb_test,2);
yMax = max(rLSE);
yMin = min(rLSE);
line([lXb_test lXb_test], [yMax yMin],'Color','green','LineStyle','--')
hold off

%% test dict on training data %%
X_test = X;
% perform omp %
D = Dksvd;
gamma = omp(D'*X_test, D'*D, k);
%ERR(D,GAMMA) = RMSE(X,D*GAMMA) = sqrt( |X-D*GAMMA|_F^2 / numel(X) )
rmse = sqrt(sum(LSE(X_test, D, gamma))/numel(X_test));

%rLSE = sqrt(sum((X_test-D*gamma).^2));
rLSE = sqrt(LSE(X_test, D, gamma));

%% show results %%
figure; plot(rLSE); title('rLSE on train set: X');
xlabel('Sample No'); ylabel('rLSE');
hold on
mmean = movmean(rLSE, 10);
plot(mmean);
lXb = size(Xb,2);
yMax = max(rLSE);
yMin = min(rLSE);
line([lXb lXb], [yMax yMin],'Color','green','LineStyle','--')
hold off


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            misc functions            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculates sum((X-D*gamma).^2) in blocks
% L2-norm loss function is also known as 
% least squares error (LSE). It is implemented by 
% minimizing the sum of the square of the differences 
function result = LSE(X, D, Gamma)
    % compute in blocks to conserve memory
    result = zeros(1,size(X,2));
    blocksize = 2000;
    for i = 1:blocksize:size(X,2)
        blockids = i : min(i+blocksize-1,size(X,2));
        result(blockids) = sum((X(:,blockids) - D*Gamma(:,blockids)).^2);
    end
end

