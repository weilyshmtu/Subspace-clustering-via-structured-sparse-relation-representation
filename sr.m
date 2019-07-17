function [Z] = sr(X,lambda)
%This routine solves the following l1-norm 
% optimization problem with l1-error
% min |Z|_1+lambda*|E|_1
% s.t., X = XZ+E
%       Zii = 0 (i.e., a data vector can not rerepsent itselft)
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
if nargin<2
    lambda = 1;
end
tol = 1e-7;
maxIter = 1e6;
[d n] = size(X);
rho = 1.1;
max_mu = 1e30;
mu = 1e-3;
xtx = X'*X;
inv_x = inv(xtx+eye(n));
%% Initializing optimization variables
% intialize
J = zeros(n);
E = sparse(d,n);
Z = J;

Y1 = zeros(d,n);
Y2 = zeros(n);
%% Start main loop
iter = 0;
while iter<maxIter
    iter = iter + 1;
    
    temp = Z + Y2/mu;
    J = max(0,temp - 1/mu)+min(0,temp + 1/mu);
    J = J - diag(diag(J)); %Jii = 0
    
    Z = inv_x*(xtx-X'*E+J+(X'*Y1-Y2)/mu);
    
    xmaz = X-X*Z;
    temp = X-X*Z+Y1/mu;
    E = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);
    
    leq1 = xmaz-E;
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end