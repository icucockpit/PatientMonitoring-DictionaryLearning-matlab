%% load data %%
clear all;
close all;

X = loadMNISTImages('train-images.idx3-ubyte'); %training signals as its columns
X = X(:,1:5000);
X_test = loadMNISTImages('t10k-images.idx3-ubyte');
X_test = X_test(:,1:5000);

% dictionary dimensions
n = size(X, 1);
m = 50;

% number of examples
L = size(X, 2);

% sparsity of each example
k = 6;

Gamma = zeros(m,L);
%% run stagewise k-svd training %%


H=2; % nb of atom updated (added-1erased, H>=2)
r=5; % ratio of iterations with full k-SVD run vs. partial K-SVD
epsilon0=4;
tB=tic;
[Dswksvd, XsprsB, DerrB, DerrMB, errIter, iterFIP, iterBIP] = trainswksvd(X, k, H, r, epsilon0);
%[Dswksvd, XsprsB, DerrB, DerrMB, errIter, iterFIP, iterBIP] = learnSWksvd(X, k, H, r, epsilon0);
timeB=toc(tB); 


%% show results %%
figure; plot(DerrB); title('K-SVD error convergence');
xlabel('Iteration'); ylabel('RMSE');

%% test dict %%

% perform omp %
D = Dswksvd;
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

