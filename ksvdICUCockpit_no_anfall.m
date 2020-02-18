%% load data %%
clear all;
close all;
disp('Loading data')
%TRAINING DATA
X = load('encoded_hist_person01.csv');
X = X';

% load labels
filename = 'encoded_hist_ann_person01.csv';
delimiter = ',';
formatSpec = '%s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
y = [dataArray{1:end-1}];

clearvars filename delimiter formatSpec fileID dataArray;

anfall_index_log = all(ismember(y,'anfall'),2);
other_index_log = all(ismember(y,'other'),2);


X_anfall = X(:,anfall_index_log);
y_anfall = y(anfall_index_log);
X_other = X(:,other_index_log);
y_other = y(other_index_log);


%% Creates splits
% no anfall
X = X_other;
y = y_other;

% Split randomly
fprintf('Random split of training set...\n');
p = .8;      % proportion to select for training
N = size(X,2);  % total number of cols 
tf = false(N, 1);    % create logical index vector
tf(1:round(p*N)) = true;    
tf = tf(randperm(N));   % randomise order
X_train = X(:,tf);
y_train = y(tf,:);
X_test = X(:,~tf);
y_test = y(~tf,:);

%% run k-svd training %%
X = X_train;

n = size(X, 1); % dictionary dimensions
m = 100; % number of atoms must be smaller than number of training signals
L = size(X, 2); % number of examples
k = 10; % sparsity of each example
Gamma = zeros(m,L);
params.data = X;
params.Tdata = k;
params.dictsize = m;
params.iternum = 1000; %number of iterations
params.memusage = 'high';
disp('Running ksvd')
[D, g, err] = ksvd(params,'');

% show results %
figure; plot(err); title('K-SVD TRAINING error convergence');
xlabel('Iteration'); ylabel('RMSE');
savefig('train_no_anfall_ksvding_error.fig');

% save dictionary %
D_train_no_anfall = D;
save('D_train_no_anfall.mat','D_train_no_anfall');

%% test dict on training data %%
weighted_sum_max = 0;
X = X_train;
% perform omp %
disp('Running omp')
gamma = omp(D'*X, D'*D, k);
GAMMA = full(gamma);
figure;
imshow(abs(GAMMA), 'InitialMagnification', 'fit', 'Colormap', copper);
colorbar;
title({'EpiKlinik p01', 'abs(gamma): TRAIN'});

% count of non-zero elements in each row
nnz_rows = sum(GAMMA~=0,2);
nnz_rows = nnz_rows / size(GAMMA,2);
% sum of non-zero elements in each row
nnz_sum = sum(abs(GAMMA), 2);
nnz_sum = nnz_sum / size(GAMMA,2);
weighted_sum = nnz_rows.*nnz_sum;
max(weighted_sum)
if (weighted_sum_max < max(weighted_sum))
    weighted_sum_max = max(weighted_sum);
end
f2 = figure;
subplot(1,3,1);
bar(weighted_sum);
title({'EpiKlinik p01', 'Weighted sum TRAIN'});
xlabel('Row no in gamma');
ylabel('Weighted sum(abs(GAMMA), 2)');
xlim([0 size(GAMMA,1)])

rLSE = sqrt(LSE(X, D, gamma));

% show results %
f1 = figure;
subplot(1,3,1);
plot(rLSE, '.');
title({'EpiKlinik p01', 'rLSE on TRAIN set'});
xlabel('Sample No');
ylabel('rLSE');
hold on
mmean = movmean(rLSE, 10);
plot(mmean);
ylim([0 150])
hold off

%% test dict on testing data %%
X = X_test;
% perform omp %
disp('Running omp')
gamma = omp(D'*X, D'*D, k);
GAMMA = full(gamma);
figure;
imshow(abs(GAMMA), 'InitialMagnification', 'fit', 'Colormap', copper);
colorbar;
title({'EpiKlinik p01', 'abs(gamma): TEST'});

% count of non-zero elements in each row
nnz_rows = sum(GAMMA~=0,2);
nnz_rows = nnz_rows / size(GAMMA,2);
% sum of non-zero elements in each row
nnz_sum = sum(abs(GAMMA), 2);
nnz_sum = nnz_sum / size(GAMMA,2);
weighted_sum = nnz_rows.*nnz_sum;
max(weighted_sum)
if (weighted_sum_max < max(weighted_sum))
    weighted_sum_max = max(weighted_sum);
end
figure(f2);
subplot(1,3,2);
bar(weighted_sum);
title({'EpiKlinik p01', 'Weighted sum TEST'});
xlabel('Row no in gamma');
ylabel('Weighted sum(abs(GAMMA), 2)');
xlim([0 size(GAMMA,1)])
savefig('test_no_anfall_weighted_sum_abs_gamma.fig');

rLSE = sqrt(LSE(X, D, gamma));

% show results %
figure(f1)
subplot(1,3,2);
hold on
plot(rLSE, '.');
title({'EpiKlinik p01', 'rLSE on TEST set'});
xlabel('Sample No'); ylabel('rLSE');
mmean = movmean(rLSE, 10);
plot(mmean);
ylim([0 150])
hold off

%% test dict on X_anfall data %%
X = X_anfall;
% perform omp %
disp('Running omp')
gamma = omp(D'*X, D'*D, k);
GAMMA = full(gamma);
figure;
imshow(abs(GAMMA), 'InitialMagnification', 'fit', 'Colormap', copper);
colorbar;
title({'EpiKlinik p01', 'abs(gamma): X\_anfall'});

% count of non-zero elements in each row
nnz_rows = sum(GAMMA~=0,2);
nnz_rows = nnz_rows / size(GAMMA,2);
% sum of non-zero elements in each row
nnz_sum = sum(abs(GAMMA), 2);
nnz_sum = nnz_sum / size(GAMMA,2);
weighted_sum = nnz_rows.*nnz_sum;
max(weighted_sum)
if (weighted_sum_max < max(weighted_sum))
    weighted_sum_max = max(weighted_sum);
end
figure(f2);
subplot(1,3,3);
bar(weighted_sum);
title({'EpiKlinik p01', 'Weighted sum X\_anfall'});
xlabel('Row no in gamma');
ylabel('Weighted sum(abs(GAMMA), 2)');
xlim([0 size(GAMMA,1)])
savefig('test_anfall_only_weighted_sum_abs_gamma.fig');

figure(f2);
subplot(1,3,1);
ylim([0 round(weighted_sum_max)])
subplot(1,3,2);
ylim([0 round(weighted_sum_max)])
subplot(1,3,3);
ylim([0 round(weighted_sum_max)])

rLSE = sqrt(LSE(X, D, gamma));
% show results %
figure(f1)
subplot(1,3,3);
hold on
plot(rLSE, '.');
title({'EpiKlinik p01', 'rLSE X\_anfall set'});
xlabel('Sample No');
ylabel('rLSE');
mmean = movmean(rLSE, 10);
plot(mmean);
ylim([0 150])
hold off

savefig('rlse_test_anfall_set.fig');

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


