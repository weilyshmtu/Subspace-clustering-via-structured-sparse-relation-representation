function [Z,E] = sclrr(X,M,beta,lambda)
% This routine solves the following nuclear-norm optimization problem 
% by using inexact Augmented Lagrange Multiplier, which has been also presented 
% in the paper entitled "Robust Subspace Segmentation 
% by Low-Rank Representation".
%------------------------------
% min |Z|_*+beta*|M.*Z|_1+lambda*|E|_2,1
% s.t., X = X*Z + E
%         ||
%         ||
%         ||
%--------------------------------
% min |J|_*+beta*|M.*W|_1+lambda*|E|_2,1
% s.t. X = X*Z + E
%      Z = J
%      Z = W
%________________________________
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.


[d n] = size(X);
%% Initializing optimization variables
% intialize the matrices
Z = zeros(n,n);
J = zeros(n,n);
W = zeros(n,n);
E = sparse(d,n);
tol = 1e-7;

Y1 = zeros(d,n);
Y2 = zeros(n,n);
Y3 = zeros(n,n);

% intialize the parameters
mu = 1e-4;
max_mu = 1e30;
rho = 1.1;

maxIter = 1e3;

xtx = X'*X;
inv_x = inv(xtx+2*eye(n));
%% Start main loop
 iter = 0;
 disp(['initial,rank=' num2str(rank(Z))]);
 while iter<maxIter
    iter = iter + 1;
 %% update J
    temp = Z + Y2/mu;
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
    %J = J-diag(diag(J));
%% update W
    temp = Z + Y3/mu;
    W = max(temp - beta/mu*M,0) + min(temp + beta/mu*M,0);
%% update Z
    Z = inv_x * (X'*(X-E+Y1/mu)+(J-Y2/mu+W-Y3/mu));
%% update E
    temp = X-X*Z+Y1/mu;
    E = solve_l1l2(temp,lambda/mu);
    %E = max(0,temp - lambda/mu) + min(0,temp + lambda/mu);
%% stop criteria        
    leq1 = X - X*Z - E;
    leq2 = Z - J;
    leq3 = Z - W;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max(stopC,max(max(abs(leq3))));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
 end

%%{
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
%} 