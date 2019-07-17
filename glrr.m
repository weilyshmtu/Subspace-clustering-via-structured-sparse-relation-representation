function [Z,E] = glrr(X,W,beta,lambda)

%This matlab code implements linearized ADM method for LRR problem
%------------------------------
% min |Z|_* +  beta * tr(Z* L*Z^T) + lambda * |E|_1
% s.t., X = XZ+E
%--------------------------------
% inputs:
%        X -- D*N data matrix
%        W -- affinity graph matrix
% outputs:
%        Z -- N*N representation matrix
%        E -- D*N sparse error matrix


%% parameters
tol = 1e-7;
maxIter = 1e6;
rho = 1.1;
max_mu = 1e30;
mu = 1e-6;

%% Initializing optimization variables
[d n] = size(X);
I = eye(n);
XtX_I = X'*X + I;

L = diag(sum(W,2)) - W;  %% Laplacian matrix

Z = zeros(n,n);
J = zeros(n,n);

Y1 = zeros(d,n);
Y2 = zeros(n,n);
E = zeros(d,n);
%% Start main loop
iter  = 0;
while iter < maxIter
    iter = iter + 1;
    %% update J
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
 %   [U,sigma,V] = lansvd(temp,30,'L');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';

    %% update Z
    tempA = mu*XtX_I;
    tempB = 4*beta*L;
    tempC = -mu*X'*(X - E +Y1/mu) - mu*(J - Y2/mu);
    Z = lyap(tempA,tempB,tempC);
    
    %% update E
    temp = X - X*Z + Y1/mu;
    E = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);
   %E = solve_l1l2(temp,lambda/mu);
    
    %%
    leq1 = X - X*Z - E;
    leq2 = Z - J;
    
    stopC = max([max(max(abs(leq1))),max(max(abs(leq2)))]);
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;

        mu = min(max_mu,mu*rho);
    end
    
end




function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end


function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end