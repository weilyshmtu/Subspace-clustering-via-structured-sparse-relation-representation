function [C,Q] = sssc(X,nbcluster,alpha,lambda)
warning off;
tol = 1e-3;    
maxIter = 1e2;
   
[Cpre] = sr(X,lambda);
%[Qpre,~] = clu_ncut(Cpre,nbcluster);
[Qpre] = spectralclustering(Cpre,nbcluster);

iter = 0;
while iter < maxIter
    iter = iter + 1;
     
    M = EuDist2(Qpre,Qpre);
    M = M.^2/2;
    [Cpost,~] = affinitymatrix(X,M,alpha,lambda);
    %[Qpost,~] = clu_ncut(Cpost,nbcluster);
    [Qpost] = spectralclustering(Cpost,nbcluster);

    
    err1 = sum(abs(Qpost(:) - Qpre(:)))/sum(abs(Qpre(:)));
    err2 = sum(abs(Cpost(:) - Cpre(:)))/sum(abs(Cpre(:)));    
    
    if iter==1 || mod(iter,5)==0 || max(err1,err2)<tol
        disp(['iter ' num2str(iter) ',err1=' num2str(err1,'%2.3e') ',err2=' num2str(err2,'%2.3e')]);
    end
    if max(err1,err2)< tol
        C = Cpost;
        Q = Qpost;
        break;
    else
        
        Qpre = Qpost;
        Cpre = Cpost;
        C = Cpost;
        Q = Qpost;
    end
        
end

end




function [C,A,E] = affinitymatrix(X,M,alpha,lambda)
% This routine is used to solve the following problem:
% min ||C||_1 + alpha*||C||_Q + lambda*||E1||_1
% s.t. X = XC + E;
%      diag(C) = 0;

%% parameters
tol = 1e-6;
maxIter = 1e5;
rho = 1.1;
max_mu = 1e30;
mu = 1e-5;% 1/min(max(X'*X,[],2));

%% Initializing optimization variables
[d,n] = size(X);
C = zeros(n,n);
A = zeros(n,n);

E = zeros(d,n);

Y1 = zeros(d,n);
Y2 = zeros(n,n);
I = eye(n);
XtX_I = X'*X + I;
%% Start main loop
iter  = 0;
while iter < maxIter
    iter = iter + 1;
    %% update C
    temp = A - 1/mu*Y2;
    C = max(0,temp - (1+alpha*M)/mu)+min(0,temp + (1+alpha*M)/mu);
    C = C - diag(diag(C));
    
    %% update A
    A = XtX_I\(X'*(X - E + Y1/mu) + C + Y2/mu);
    %A = A - diag(diag(A));
    %% update E
    temp = X - X*A + Y1/mu;
    E = max(0,temp - lambda/mu) + min(0,temp + lambda/mu);

    %% update Langrange multipliers
    leq1 = X - X*A - E;
    leq2 = C - A;
    
    stopC = max([max(max(abs(leq1))),max(max(abs(leq2)))]);
   % if iter==1 || mod(iter,50)==0 || stopC<tol
   %     disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
   %         ',rank=' num2str(rank(C,1e-3*norm(C,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
   % end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end

end
end

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
    %if iter==1 || mod(iter,50)==0 || stopC<tol
    %    disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ',stopALM=' num2str(stopC,'%2.3e')]);
    %end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end
end


function [H] = spectralclustering(A,K)
A = 0.5*(abs(A) + abs(A'));
D = diag(sum(A,2));
L = D - A;
%M = sqrt(D^(-1))*L*sqrt(D^(-1));
M = L;
%options.disp = 0; options.isreal = 1; options.issym = 0; 
[U,S,V] = svd(M);
U = U(:,end-K:end);
H = normr(U);
%H = kmeans(V,K,'emptyaction','singleton','replicates',100,'display','off');

end


function D = EuDist2(fea_a,fea_b,bSqrt)
% Euclidean Distance matrix
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b


if ~exist('bSqrt','var')
    bSqrt = 1;
end


if (~exist('fea_b','var')) | isempty(fea_b)
    [nSmp, nFea] = size(fea_a);

    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    aa = full(aa);
    ab = full(ab);

    if bSqrt
        D = sqrt(repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab);
        D = real(D);
    else
        D = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
    end
    
    D = max(D,D');
    D = D - diag(diag(D));
    D = abs(D);
else
    [nSmp_a, nFea] = size(fea_a);
    [nSmp_b, nFea] = size(fea_b);
    
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    aa = full(aa);
    bb = full(bb);
    ab = full(ab);

    if bSqrt
        D = sqrt(repmat(aa, 1, nSmp_b) + repmat(bb', nSmp_a, 1) - 2*ab);
        D = real(D);
    else
        D = repmat(aa, 1, nSmp_b) + repmat(bb', nSmp_a, 1) - 2*ab;
    end
    
    D = abs(D);
end
end