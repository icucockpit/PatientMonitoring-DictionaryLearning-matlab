%% load data %%
clear all;
close all;
disp('Loading data')

%TRAINING DATA
X2 = load('icu_4000\mrn5805951_180129_000001_anfall_02_tip_hog.csv_w101_s20_histenc_ts.csv');
% X3 = load('icu_4000\mrn5805951_180129_000001_anfall_03_tip_hog.csv_w101_s20_histenc_ts.csv');
X4 = load('icu_4000\mrn5805951_180129_000001_anfall_04_tip_hog.csv_w101_s20_histenc_ts.csv');
X5 = load('icu_4000\mrn5805951_180129_000001_anfall_05_tip_hog.csv_w101_s20_histenc_ts.csv');
X6 = load('icu_4000\mrn5805951_180129_000001_anfall_06_tip_hog.csv_w101_s20_histenc_ts.csv');

%X = [X2', X3', X4', X5', X6'];
X = [X2', X4', X5', X6'];

% load labels
delimiter = ',';
formatSpec = '%s%s%s%[^\n\r]';
filename = 'icu_4000\mrn5805951_180129_000001_anfall_02_tip_hog.csv_w101_s20_histenc_ann_ts.csv';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
y2 = [dataArray{1:end-1}];

% filename = 'icu_4000\mrn5805951_180129_000001_anfall_03_tip_hog.csv_w101_s20_histenc_ann_ts.csv';
% fileID = fopen(filename,'r');
% dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
% fclose(fileID);
% y3 = [dataArray{1:end-1}];

filename = 'icu_4000\mrn5805951_180129_000001_anfall_04_tip_hog.csv_w101_s20_histenc_ann_ts.csv';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
y4 = [dataArray{1:end-1}];
filename = 'icu_4000\mrn5805951_180129_000001_anfall_05_tip_hog.csv_w101_s20_histenc_ann_ts.csv';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
y5 = [dataArray{1:end-1}];
filename = 'icu_4000\mrn5805951_180129_000001_anfall_06_tip_hog.csv_w101_s20_histenc_ann_ts.csv';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);
fclose(fileID);
y6 = [dataArray{1:end-1}];

% y = [y2; y3; y4; y5; y6];
y = [y2; y4; y5; y6];

%x_signal = X(1,:);
%y_datetime = y(:,1);
%y_datetime_matlab = datetime(y_datetime, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');
%plot(y_datetime_matlab, x_signal, '.')

clearvars filename delimiter formatSpec fileID dataArray;
clearvars y2 y3 y4 y5 y6;
clearvars X2 X3 X4 X5 X6;

anfall_index_log = all(ismember(y(:,3),'ANFALL'),2);
other_index_log = all(ismember(y(:,3),'no_label'),2);


X_anfall = X(:,anfall_index_log);
y_anfall = y(anfall_index_log, 3);
ts_anfall = y(anfall_index_log, 1);
ts_anfall_datetime = datetime(ts_anfall, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');

X_other = X(:,other_index_log);
y_other = y(other_index_log, 3);
ts_other = y(other_index_log, 1);


%% Creates splits
% no anfall
X = X_other;
y = y_other;
ts = ts_other;

% Split randomly
fprintf('Random split of training set...\n');
p = .8;      % proportion to select for training
N = size(X,2);  % total number of cols 
tf = false(N, 1);    % create logical index vector
tf(1:round(p*N)) = true;    
tf = tf(randperm(N));   % randomise order
X_train = X(:,tf);
y_train = y(tf,:);
ts_train = ts(tf,:);
ts_train_datetime = datetime(ts_train, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');

X_test = X(:,~tf);
y_test = y(~tf,:);
ts_test = ts(~tf,:);
ts_test_datetime = datetime(ts_test, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSSSSS');

save('X_train.mat','X_train')
save('X_test.mat','X_test')
save('X_anfall.mat','X_anfall')

save('y_train.mat','y_train')
save('y_test.mat','y_test')
save('y_anfall.mat','y_anfall')

save('ts_train.mat','ts_train')
save('ts_test.mat','ts_test')
save('ts_anfall.mat','ts_anfall')


%% run k-svd training %%
X = X_train;

n = size(X, 1); % dictionary dimensions
m = 1000; % number of atoms must be smaller than number of training signals
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
title({'TRAIN'});

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
title({'TRAIN'});
xlabel('Row no. in gamma');
ylabel('Weighted sum(abs(GAMMA), 2)');
xlim([0 size(GAMMA,1)])

rLSE = sqrt(LSE(X, D, gamma));

% show results %
f1 = figure;
subplot(1,3,1);
plot(ts_train_datetime, rLSE, '.');
title({'TRAIN'});
xlabel('Sample No');
ylabel('rLSE');
hold on
mmean = movmean(rLSE, 10);
plot(ts_train_datetime, mmean);
%xlim([0 size(rLSE,2)])
%xlim([0 100])
ylim([0 0.1])
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
title({'TEST'});

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
title({'TEST'});
xlabel('Row no. in gamma');
ylabel('Weighted sum(abs(GAMMA), 2)');
xlim([0 size(GAMMA,1)])
savefig('test_no_anfall_weighted_sum_abs_gamma.fig');

rLSE = sqrt(LSE(X, D, gamma));

% show results %
figure(f1)
subplot(1,3,2);
hold on
plot(ts_test_datetime, rLSE, '.');
title({'TEST'});
xlabel('Sample No'); ylabel('rLSE');
mmean = movmean(rLSE, 10);
plot(ts_test_datetime, mmean);
%xlim([0 size(rLSE,2)])
%xlim([0 100])
ylim([0 0.1])
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
title({'EVENT'});

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
title({'EVENT'});
xlabel('Row no. in gamma');
ylabel('Weighted sum(abs(GAMMA), 2)');
xlim([0 size(GAMMA,1)])
savefig('test_anfall_only_weighted_sum_abs_gamma.fig');

figure(f2);
subplot(1,3,1);
ylim([0 weighted_sum_max])
subplot(1,3,2);
ylim([0 weighted_sum_max])
subplot(1,3,3);
ylim([0 weighted_sum_max])

rLSE = sqrt(LSE(X, D, gamma));
% show results %
figure(f1)
subplot(1,3,3);
hold on
plot(ts_anfall_datetime, rLSE, '.');
title({'EVENT'});
xlabel('Sample No');
ylabel('rLSE');
mmean = movmean(rLSE, 10);
plot(ts_anfall_datetime, mmean);
%xlim([0 size(rLSE,2)])
%xlim([0 100])
ylim([0 0.1])
hold off


figure(f1);
subplot(1,3,1);
ylim([0 0.2])
subplot(1,3,2);
ylim([0 0.2])
subplot(1,3,3);
ylim([0 0.2])



savefig('rlse_test_anfall_set.fig');

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


