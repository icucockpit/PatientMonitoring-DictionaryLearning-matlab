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