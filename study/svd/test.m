digits(16);
r = 500;  m = 50;
r1 = r;
r2 = r;
r3 = r;
x1 = single(rand(r1,m));
x2 = single(rand(r2,m));
x3 = single(rand(r3,m));

nR_F=size(x1,1);        %FµÄÐÐ
nR_G=size(x2,1);      %GµÄÐÐÊý
mul=ones(nR_G,1);
FF=kron(x1,mul);     %Í¨¹ýkronº¯ÊýÊµÏÖ¶ÔF¾ØÕóµÄÀ©³ä£¬µÃµ½FF¾ØÕó
GG=repmat(x2,nR_F,1);%Í¨¹ýrepmatº¯ÊýÊµÏÖ¶ÔG¾ØÕóµÄÀ©³ä£¬µÃµ½GG¾ØÕó
kr=FF.*GG;

kr = reshape(kr,[numel(kr)/m,m]);
A = kr * x3';%'
A = reshape(A,[numel(A)/r,r]);
niter = 5;
k = 50;
step = 10;
p = 1; % This can be modified for different power iteration
Anorm = norm(A, 'fro');
eps = 1e-6;
fprintf('Finish general A\n');
for p = 1:1
%     profile on
    fprintf('p = %d\n', p);
    time = 0;
%     profile on
    for i = 1:niter   
        k = 0;
        epsRe = 7.0194e+04;
        t = cputime;
    	while ( epsRe - eps > 0 )
        	[U0, S0, V0] = basicrSVD(A, k, p);
            epsRe = norm(A - U0*S0*V0') / norm(A);
        	k = k + step;
        end
        time = time + cputime - t;
        fprintf('k = %d\n', k);
    end
%     profile viewer
%     p = profile('info');
%     profsave(p,'profile_results') % ±£´æprofile ½á¹û
    time_0 = time/niter
    err_0 = norm(A-U0*S0*V0', 'fro')/Anorm %'
end
% fprintf('classical svd\n');
% time = 0;
% for i = 1:niter
%     t = cputime;
%     [U0, S0, V0] = svd(A, 'econ');
%     time = time + cputime - t;
% end
% time_1 = time/niter