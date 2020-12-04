A = rand(16,2);
[Q,R]=qr(A,0);
Acell = cell(1,4);
Q1 = cell(1,4);
R1 = cell(1,4);
for i=1:4
    Acell{i}=A(i*4-3:i*4,:);
    [Q1{i} R1{i}] = qr(Acell{i},0);
end
Acell2 = cell(1,2);
Q2 = cell(1,2);
R2 = cell(1,2);
for i=1:2
    Acell2{i}=[R1{i*2-1};R1{i*2}];
    [Q2{i} R2{i}] = qr(Acell2{i},0);
end
[Q3 R3] = qr([R2{1};R2{2}],0);
Qr2=blkdiag(Q2{1},Q2{2});
Qr1=blkdiag(Q1{1},Q1{2},Q1{3},Q1{4});
Qr = Qr1*Qr2*Q3;
assert(norm(Qr-Q)/norm(Q) < 10^-5);
assert(norm(R3-R)/norm(R) < 10^-5);