%% load data %%
clear all;
close all;

X = loadMNISTImages('train-images.idx3-ubyte'); %training signals as its columns
X = X(:,1:5000);
X_test = loadMNISTImages('t10k-images.idx3-ubyte');
X_test = X_test(:,1:5000);

% dictionary dimensions
n = size(X, 1);
m = 1000; %number of atoms

% number of examples
L = size(X, 2);

% sparsity of each example
k = 3;

Gamma = zeros(m,L);
%% run k-svd training %%

params.data = X;
params.Tdata = k;
params.dictsize = m;
params.iternum = 30;
params.memusage = 'high';

[Dksvd, g, err] = ksvd(params,'');

%% show results %%
figure; plot(err); title('K-SVD error convergence');
xlabel('Iteration'); ylabel('RMSE');

%% test dict %%

% perform omp %
D = Dksvd;
gamma = omp(D'*X_test, D'*D, k);
%ERR(D,GAMMA) = RMSE(X,D*GAMMA) = sqrt( |X-D*GAMMA|_F^2 / numel(X) )
rmse = sqrt(sum(LSE(X_test, D, gamma))/numel(X_test));

%rLSE = sqrt(sum((X_test-D*gamma).^2));
rLSE = sqrt(LSE(X_test, D, gamma));

%% show results %%
figure; plot(rLSE); title('rLSE on test set');
xlabel('Sample No'); ylabel('rLSE');

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

