function [u, v, d, iter] = SSVD(X,k_u,k_v,J,thr,maxit)

% SSVD  Computes sparse singular vectors
%   [Q, log] = SSVD(X, k_u, k_v, J, thr, maxit) computes a pair of sparse
%   singular vectors of the rectangular matrix X, each having at most k_u 
%   or k_v non-zero indices.
%
%   INPUTS:
%       X:      Data matrix
%       k_u:    Maximum number of non-zero indices in u
%       k_v:    Maximum number of non-zero indices in v
%       J:      Number of power method steps to be taken
%       maxit:  Maximum number of iterations to take
%       thr:    Accuracy threshold
%
%   OUTPUTS:
%       u:      Left singular value
%       v:      Right singular value
%       d:      Variance explained
%       iter:   Number of iterations taken
%
%   Singular vectors are computed using a technique called Generalized
%   Rayleigh quotient iteration. At every iteration, the non-zero indices
%   are updated using Rayleigh quotient iteration. For the first J
%   iterations, every index is also updates using a step of the power
%   method. Afterwards, the iterate is projected on the set defined by 
%   the sparsity constraints.
%
%   In order to handle rectangular matrices, Generalized Rayleigh quotient
%   iteration is applied on the symmetric matrix Y = [0 X'; X 0]. However, 
%   Y is never explicitely formed. Instead, we perform inversions on
%   submatrices of Y using the matrix inversion lemma.
%
%   For more information on the method see the paper
%
%   V. Kuleshov, Fast algorithms for sparse principal component Analysis 
%   based on Rayleigh quotient iteration. Proceedings of the 30th 
%   International Conference on Machine Learning, Atlanta, GA, 2013.
    
% First, initialize u_0, v_0, mu

column_norms = sqrt(sum(X.^2,1));
[~, idx] = max(column_norms);
u = X(:,idx)/norm(X(:,idx));
u = l0_project(u,k_u);

row_norms = sqrt(sum(X.^2,2));
[~, idx] = max(row_norms);
v = (X(idx,:)/norm(X(idx,:)))';
v = l0_project(v,k_v);

mu = u'*X*v/(norm(u)*norm(v));

err = 1; iter = 0;

while err > thr && iter < maxit;

    oldu = u;
    oldv = v;

    % Compute working sets

    Wo_u = find(u ~= 0);
    Wo_v = find(v ~= 0);

    % Perform a step of Rayleigh quotient iteration of the working set

    A = X(Wo_u,Wo_v);
    [m, ~] = size(A);
    u_Wo = u(Wo_u);
    v_Wo = v(Wo_v);

    % We now invert [0 A'; A 0] - mu*I using the matrix inversion
    % lemma:
    %
    % B = [-mu*eye(n) A'; A -mu*eye(m)];
    % Binv = [(1/mu^2)*A'*inv(S)*A - (1/mu)*eye(n) (1/mu)*A'*inv(S);
    %          inv(S)*A inv(S);]
    %
    % where S is the Schur complement:
    S = (A*A')/mu - mu*eye(m);

    % (1,1) block
    Av = A*v_Wo;
    SiAv = S \ Av;
    AtSiAv = A'*SiAv;
    v_part1 = AtSiAv / (mu^2) - v_Wo/mu;

    % (1,2) block
    Siu = S \ u_Wo;
    AtSiu = A'*Siu;
    v_part2 = AtSiu / mu;
    v(Wo_v) = v_part1 + v_part2;

    % (2,1) block
    u_part1 = SiAv / mu;

    % (2,2) block
    u_part2 = Siu;

    u(Wo_u) = u_part1 + u_part2;
    mu = u'*X*v/(norm(u)*norm(v));

    u = u / norm(u);
    v = v / norm(v);

    % Perform a step of the Power method on all indices
    if iter < J
        u = X*v;
        v = X'*u;
    end

    % Project on the intersection of the l0 and l2 balls

    u = l0_project(u,k_u);
    v = l0_project(v,k_v);

    [~, n] = size(X);
    x = [v; u];
    x = x/norm(x);
    v = x(1:n);
    u = x(n+1:end);

    erru = norm(oldu-u,2);
    errv = norm(oldv-v,2);
    err = erru + errv;

    variance = u'*X*v/(norm(v)*norm(u));

    fprintf('%d: %f \t%f\n',iter,err,variance);

    iter = iter + 1;
end

d = variance;

fprintf('RESULTS:\nVariance: %f\n\tSparsity: %d, %d\n',...
    variance ,nnz(u), nnz(v));
    
end

function x = l0_project(x, k)
    [~, idx] = sort(abs(x),'descend');
    idx_to_zero = idx(k+1:end);
    x(idx_to_zero) = 0;
    x = x / norm(x);
end
