function [C,Z,E1,E2] = ssrr1(X,alpha,beta)

% This routine is used to solve the following problem:
% min ||M.*C||_1 + ||Z||_* +alpha||E1||_1 + beta||E2||_F^2
% s.t. X = XC + E1
%      C = CZ + E2

%% parameters
tol = 1e-6;
maxIter = 1e5;
rho = 1.1;
max_mu = 1e30;
mu = 1e-2;

%% Initializing optimization variables
[d,n] = size(X);
C = zeros(n,n);
W = zeros(n,n);
Z = zeros(n,n);
J = zeros(n,n);
M = ones(n,n);

E1 = zeros(d,n);
E2 = zeros(n,n);

Y1 = zeros(d,n);
Y2 = zeros(n,n);
Y3 = Y2;
Y4 = Y2;
I = eye(n);
XtX_I = X'*X + I;
%% Start main loop
iter  = 0;
while iter < maxIter
    iter = iter + 1;
    %% update W
    temp = C + Y2/mu;
    W = max(0,temp - M/mu)+min(0,temp + M/mu);
    W = W - diag(diag(W));
    %% update C
    tempA = XtX_I;
    tempB = I - Z' - Z + Z*Z';
    tempC = -X'*(X - E1 + Y1/mu) - W + Y2/mu - (E2 - Y3/mu)*(I - Z');
    C = lyap(tempA,tempB,tempC);
    
    %% update J
    temp = Z + Y4/mu;
    [U,sigma,V] = svd(temp,'econ');
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
    Z = (C'*C + I)\(C'*(C - E2 +Y3/mu) + J - Y4/mu);
    M = Eudist2(Z',Z');
    %% update E1
    temp = X - X*C +Y1/mu;
    E1 = max(0,temp - alpha/mu)+min(0,temp + alpha/mu);
    %% update E2
    temp = C - C*Z + Y3/mu;
    E2 = mu/(2*beta +mu)*temp;
    %%
    leq1 = X - X*C - E1;
    leq2 = C - W; 
    leq3 = C - C*Z - E2; 
    leq4 = Z - J;
    
    stopC = max([max(max(abs(leq1))),max(max(abs(leq2))),max(max(abs(leq3))),max(max(abs(leq4)))]);
    
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        Y4 = Y4 + mu*leq4;
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
