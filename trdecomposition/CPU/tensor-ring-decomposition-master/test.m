digits(16)

r = 1200
m = 50;
r1 = r;
r2 = r;
r3 = r;
x1 = single(rand(r1,m));
x2 = single(rand(r2,m));
x3 = single(rand(r3,m));

nR_F=size(x1,1);        %F的行
nR_G=size(x2,1);      %G的行数
mul=ones(nR_G,1);
FF=kron(x1,mul);     %通过kron函数实现对F矩阵的扩充，得到FF矩阵
GG=repmat(x2,nR_F,1);%通过repmat函数实现对G矩阵的扩充，得到GG矩阵
kr=FF.*GG;

kr = reshape(kr,[numel(kr)/m,m]);
x = kr * x3'; %'
x = reshape(x,[r1,r2,r3]);
timeall = 0.0;
caltime = 5

for i=1:caltime
    tic;
    t1 = toc;
   	q = TRdecomp(x, 1e-6, 2)
    t2 = toc;
    timeall = timeall + t2 - t1;
end
timeall/caltime
