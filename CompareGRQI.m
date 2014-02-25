%%% This script compares GRQI vs. GPower on random matrices.

randn('seed',1)
n = 500;

A = randn(n,n);
A = A'*A;

[~, log0] = GRQI(A,44,1,Inf,0,150,1e-6);
[~, log1] = GPower(A,1.7,1,0,150,1e-6);

log01 = log0{1};
log11 = log1{1};

nm_errors = log0{1}.errors;
nm_variances = log0{1}.variances;
nm_sparsities = log0{1}.sparsities;

power_errors = log1{1}.errors;
power_variances = log1{1}.variances;
power_sparsities = log1{1}.sparsities;

subplot(3,2,1);
plot(nm_errors);
t = title('Convergence rate (GRQI)', 'FontWeight','bold');
set(t, 'FontSize', 11);
xlabel('Iterations')
ylabel('|| x - x_{prev} ||')
xlim([1,6])
ylim([0,0.8])

subplot(3,2,2);
plot(power_errors);
t = title('Convergence rate (GPower)', 'FontWeight','bold');
set(t, 'FontSize', 11);
xlabel('Iterations')
ylabel('|| x - x_{prev} ||')
xlim([0 150]);

subplot(3,2,3);
plot(nm_variances);
t = title('Variance (GRQI)', 'FontWeight','bold');
set(t, 'FontSize', 11);
xlabel('Iterations')
ylabel('Variance')
xlim([1,6])

subplot(3,2,4);
plot(power_variances);
t = title('Variance (GPower)', 'FontWeight','bold');
set(t, 'FontSize', 11);
xlabel('Iterations')
ylabel('Variance')
xlim([0 150]);

subplot(3,2,5);
plot(nm_sparsities);
t = title('Sparsity (GRQI)', 'FontWeight','bold');
set(t, 'FontSize', 11);
xlabel('Iterations')
ylabel('Number of non-zero entries')
xlim([1,6])

subplot(3,2,6);
plot(power_sparsities);
t = title('Sparsity (GPower)', 'FontWeight','bold');
set(t, 'FontSize', 11);
xlabel('Iterations')
ylabel('Number of non-zero entries')
xlim([0 150]);
ylim([35, 60]);