function [Q, full_log] = GPower(X,rho,K,alpha,maxit,thr)

% GRQI  Computes sparse principal components
%   [Q, log] = GRQI(X, k, K, J, alpha, maxit, thr) computes K sparse
%   principal components of symmetric matrix X.
%
%   INPUTS:
%       X:      Data matrix
%       rho:    Regularization parameter
%       K:      Number of sparse principal componenents to compute
%       alpha:  Deflation parameter
%       maxit:  Maximum number of iterations to take
%       thr:    Accuracy threshold
%
%   OUTPUTS:
%       Q:      Matrix of sparse principal componenents
%       log:    Runtime stats
%
%   Principal componenets are computed using a technique called the 
%   Generalized Power Method For more information on the method see
%
%   M. Journee, Y. Nesterov, P. Richtarik, R. Sepulchre, Generalized power 
%   Method for sparse principal component analysis, arXiv:0811.4724v1, 2008

n = size(X,1);
Q = zeros(n,K);
full_log = cell(K,1);

% This is necessary for comparison to GRQI. GPower assumes that the
% diagonal matrix X is factored as X = A'*A. In typical scenarios, X is
% going to be the covariance matrix and A will be the data matrix.
A = chol(X);
assert(all(all(A'*A - X < 1e-4)));

for i=1:K
    % Log statistics for this run:
    log = struct('errors', [], 'variances', [], 'sparsities', []);
    
    % First, initialize x_0

    column_norms = sqrt(sum(X.^2,1));
    [~, col_i] = max(column_norms);
    x = X(:,col_i)/norm(X(:,col_i));
    
    err = 1; iter = 0;
    disp(i);
        
    while err > thr && iter<maxit;
        oldx = x;

        % The l0/l1 updates below are taken directly from the source code
        % of Journee et al. (2008)
        
        % l1 update:
        Ax=A'*x;
        tresh=sign(Ax).*max(abs(Ax)-rho,0);
        grad=A*tresh;
        x=grad/norm(grad);
        
        var = tresh'*X*tresh/(tresh'*tresh);

        % l0 update:
%         Ax=A'*x;
%         tresh=max(Ax.^2-rho,0);
%         grad=A*((tresh>0).*Ax);
%         x=grad/norm(grad);   
                    
        err = norm(x-oldx);
        
        % Save run statistics
        log.errors = [log.errors err];
        log.variances = [log.variances var];
        log.sparsities = [log.sparsities nnz(tresh)];
        
        iter = iter + 1;
        
        fprintf('%d \t %d: %f \t %f \t %d\n', i, iter, err, var, ...
                                                    nnz(tresh));
    end
    
    Ax=A'*x;
    z=sign(Ax).*max(abs(Ax)-rho,0);
    if max(abs(z>0))>0, 
        z=z/norm(z);
    end
    x=z;
    
    % Save run statistics
    log.errors = [log.errors err];
    log.variances = [log.variances var];
    log.sparsities = [log.sparsities nnz(tresh)];
    full_log{i} = log;
   
    fprintf('RESULTS:\n\tPrecision: %f\n\tVariance: %f\n\tSparsity: %d\n',...
        err, var, nnz(x));
    
    Q(:,i) = x;
    
    % Perform partial deflation
    X = X - alpha*var*x*x';
end
end