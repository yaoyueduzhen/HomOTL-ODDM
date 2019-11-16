function  [mistakes_idx] =  Experiment(source_domains, target_domain)
% Experiment: the main function used to run HomOTL-ODDM.
% 
%--------------------------------------------------------------------------
% Input:
%      source_domains: a set of source dataset_name, e.g. {'PIE1.mat','PIE2.mat','PIE3.mat','PIE4.mat'}
%      target_domain: target dataset_name, e.g. 'PIE5.mat'
%--------------------------------------------------------------------------

data_nums = [];
%load dataset
train_data = [];
for i = 1:length(source_domains),
    load(sprintf('data/%s', source_domains{i}));
    train_data = [train_data; data];
    data_nums = [data_nums;size(data,1)];
end

load(sprintf('data/%s', target_domain));
test_data = data;

m = size(test_data,1);
num = int32(m*0.3);
ID_new=[];
for i=1:20,
    ID_new = [ID_new; randperm(m-num)];
end

% options
options.C      = 5;
options.t_tick = 50;
options.k = labels_num; 
options.dim = 100;
options.mu = 1;

%
m = length(ID_new);
options.beta1 = sqrt(m)/(sqrt(m)+sqrt(log(2)));
options.beta2 = sqrt(m)/(sqrt(m)+sqrt(log(length(source_domains) + 1)));

%JDA pretrain
JDAoption.lambda = 0.1;              
JDAoption.dim = 100;                    
JDAoption.kernel_type = 'primal';    
JDAoption.gamma = 4;               
JDAoption.T = 10;

%% run experiments:

hs = [];
for i = 1:length(source_domains)
    [h,A,X,Y,X2,mean_Xs, mean_Xt, num, mean_ks, mean_kt, num_kt] = source_classifier(source_domains{i},target_domain, options,JDAoption);
    hs = [hs,h];  %hs(i).W  soure domain classifier 
    unlabeled_data{i} = X2;  %unlabeled target data
    MEAN_Xs{i} = mean_Xs;  %mean of source data
    MEAN_Xt{i} = mean_Xt;  %mean of target data
    NUM_t{i} = double(num);  %number of target data
    MEAN_ks{i} = mean_ks;   %mean of source data whose label are k
    MEAN_kt{i} = mean_kt;   %mean of target data whose label are k, initialized to 0
    NUM_kt{i} = double(num_kt); %number of target data whose label are k, initialized to 0

    JDA_A{i} = A;  %transformation matrix
    JDA_X{i} = X;  %target data
    JDA_Y{i} = Y;  %target label
end


for i=1:size(ID_new,1),
    fprintf(1,'running on the %d-th trial...\n',i);
    ID = ID_new(i, :);
    
    % HomOTL-ODDM
    [classifier, err_count, run_time, mistakes, mistakes_idx] = HomOTL_ODDM(JDA_Y,JDA_A,JDA_X,MEAN_Xs,MEAN_Xt,NUM_t,MEAN_ks,MEAN_kt,NUM_kt,options,ID,hs);
    err_ODDM(i) = err_count;
    time_ODDM(i) = run_time;
    mistakes_list_ODDM(i,:) = mistakes;
end

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'number of mistakes\n');
fprintf(1,'HomOTL-ODDM   %.4f \t %.4f \n', mean(err_ODDM)/m*100,   std(err_ODDM)/m*100);
fprintf(1,'-------------------------------------------------------------------------------\n');

