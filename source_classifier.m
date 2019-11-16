function [ h, A, X, Y, Z2,mean_Xs, mean_Xt, num, mean_ks, mean_kt, num_kt] = source_classifier(source_domain,target_domain, options,JDAoption)

%load dataset
load(sprintf('data/%s', source_domain));
source_data = data(:,:);
ID = ID_old;


load(sprintf('data/%s', target_domain));
target_data = data(ID_old,:);

data = [source_data;target_data];
[n,d]       = size(data);
m = size(target_data,1);
Y=data(1:n,1);
Y=full(Y);
X = data(1:n,2:d);

% scale
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);

X = X';
X = X*diag(sparse(1./sqrt(sum(X.^2))));
X = X';

num = int32(m*0.3);
X2 = X(n-m+1:n-m+num,:);  %online data
Y2 = Y(n-m+1:n-m+num);

mean_Xs = mean(X(1:n-m,:));   %mean of source data
mean_Xt = mean(X(n-m+1:n-m+num,:));  %mean of unlabeled target data

num_ks = zeros(options.k,1);
sum_ks = zeros(options.k,d-1);

for i = 1:n-m
    num_ks(Y(i)) = num_ks(Y(i)) + 1;
    sum_ks(Y(i),:) = sum_ks(Y(i),:) + X(i,:);
end

for i = 1:options.k
    mean_ks(i,:) = sum_ks(i,:)./num_ks(i);
end

num_kt = zeros(options.k,1);
sum_kt = zeros(options.k,d-1);
mean_kt = zeros(options.k,d-1);

[acc,acc_ite,A,Z] = JDA(X(1:n-m,:),Y(1:n-m),X2,Y2,JDAoption);

Z2 = Z(:,n-m+1:n-m+num)';

[h, err_count, run_time, mistakes, mistakes_idx, TMs] = avePA1_K_M(Y, Z', options, ID);

X = X(n-m+num+1:n,:);
Y = Y(n-m+num+1:n);
end

