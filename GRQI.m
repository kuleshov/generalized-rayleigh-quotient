function [Q, full_log] = GRQI(X,k,K,J,alpha,maxit,thr)

% GRQI  Computes sparse principal components
%   [Q, log] = GRQI(X, k, K, J, alpha, maxit, thr) computes K principal
%   components of symmetric matrix X, each having at most k non-zero
%   indices.
%
%   INPUTS:
%       X:      Data matrix
%       k:      Maximum number of non-zero indices
%       K:      Number of sparse principal componenents to compute
%       J:      Number of power method steps to be taken
%       alpha:  Deflation parameter
%       maxit:  Maximum number of iterations to take
%       thr:    Accuracy threshold
%
%   OUTPUTS:
%       Q:      Matrix of sparse principal componenents
%       log:    Runtime stats
%
%   Principal componenets are computed using a technique called Generalized
%   Rayleigh quotient iteration. At every iteration, the non-zero indices
%   are updated using Rayleigh quotient iteration. For the first J
%   iterations, every index is also updates using a step of the power
%   method. Afterwards, the iterate is projected on the set defined by 
%   ||x||_0 <= k  and ||x||_2 <= 1.
%
%   After computing each principal component, it deflates the matrix X by 
%   removing a fraction alpha of the variance explained by last component.
%
%   For more information on the method see the paper
%
%   V. Kuleshov, Fast algorithms for sparse principal component Analysis 
%   based on Rayleigh quotient iteration. Proceedings of the 30th 
%   International Conference on Machine Learning, Atlanta, GA, 2013.

n = size(X,1);
Q = zeros(n,K);
full_log = cell(K,1);

for i=1:K
    % Log statistics for this run:
    log = struct('errors', [], 'variances', [], 'sparsities', []);
    
    % First, initialize x_0
    
    % We recommend taking the largest column of the input matrix:
    column_norms = sqrt(sum(X.^2,1));
    [~, col_i] = max(column_norms);
    x = X(:,col_i);
    x = l0_project(x,k);
    mu = x'*X*x/(x'*x);
    
    % Another option is to initalize randomly, and use a mu that is close
    % to the largest eigenvalue. The value of mu can be guessed or
    % computed.    
    % x = randn(n,1); x = x/norm(x);
    % mu = 1900;

    err = 1; iter = 0;
    disp(i);
        
    while err > thr && iter < maxit;
        oldx = x;
        
        % Compute working set
        Wo = find(x~=0);
        
        % Perform a Rayleigh quotient iteration update across the working
        % set.
        A = X(Wo,Wo);
        
        % The code below is a more numerically stable way of doing the
        % update:
        % x(Wo) = (A-mu*eye(size(A))) \ x(Wo);
        % It performs a step of Newton's method on the KKT conditions of
        % the problem max x'*A*x s.t. 0.5*x'*x == 1.
        % One can check using the matrix inversion theorem that they 
        % ultimately produce the same update to x(Wo).
        
        G = -(A-mu*eye(size(A)));
        DF = [G, x(Wo); x(Wo)', 0];
        F = [G*x(Wo); 0.5*(x(Wo)'*x(Wo) - 1)];
        delta = DF \ -F;
        x(Wo) = x(Wo) + delta(1:end-1);
        
        % Update mu and renormalize
        mu = x(Wo)'*X(Wo,Wo)*x(Wo)/(x(Wo)'*x(Wo));
        x = x/norm(x);

        % Perform a Power method step across all indices
        if (iter < J)
            x = X*x;
        end

        % Project on the intersection of the l2 and l0 balls
        x = l0_project(x,k);

        err = norm(oldx - x,2);
        variance = x'*X*x/(x'*x);
        
        iter = iter + 1;
        
        % Save run statistics
        log.errors = [log.errors err];
        log.variances = [log.variances variance];
        log.sparsities = [log.sparsities nnz(x)];
        
        errors = log.errors;
        variances = log.variances;
        sparsities = log.sparsities;
        
        % Print current state
        fprintf('%d \t %d: %f \t %f \t %d\n', i, iter, err, ...
                                              variance, nnz(x));
    end
    
    log.errors = [log.errors err];
    log.variances = [log.variances variance];
    log.sparsities = [log.sparsities nnz(x)];
    full_log{i} = log;
   
    fprintf('RESULTS:\n\tPrecision: %f\n\tVariance: %f\n\tSparsity: %d\n',...
        err, variance, nnz(x));
    
    Q(:,i) = x;
    
    % Perform partial deflation
    X = X - alpha*variance*x*x';
     
end
end

function x = l0_project(x, k)
    [~, idx] = sort(abs(x),'descend');
    idx_to_zero = idx(k+1:end);
    x(idx_to_zero) = 0;
    x = x / norm(x);
end