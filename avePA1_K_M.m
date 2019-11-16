function [classifier, err_count, run_time, mistakes, mistakes_idx, TMs] = avePA1_K_M(Y, X, options, id_list)
% avePA1: Averge online passive-aggressive algorithm
%--------------------------------------------------------------------------
% Input:
%        Y:    the vector of lables
%  id_list:    a randomized ID list
%  options:    a struct containing rho, sigma, C, n_label and n_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm once
%    mistakes:  a vector of mistake rate 
% mistake_idx:  a vector of number, in which every number corresponds to a
%               mistake rate in the vector above
%--------------------------------------------------------------------------

%% initialize parameters
C = options.C; % 5 by default
t_tick = options.t_tick;
k = options.k;
W = zeros(k,size(X,2));
ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [];
TMs=[];
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    x_t = X(id,:);
    y_t = Y(id);
    
    F_t = W*x_t';
    [F_max,hat_y_t]=max(F_t);
    
    Fs=F_t;
    Fs(y_t)=-inf;
    [Fs_max, s_t]=max(Fs);
    l_t = max(0, 1 - (F_t(y_t) - F_t(s_t))); 
    %--------------------------------------------------------------------------
    % Making Update
    %--------------------------------------------------------------------------    
    if (hat_y_t~=y_t),
        err_count = err_count + 1;
    end
    
    if (l_t > 0),
        eta_t = min(C, l_t/(2*norm(x_t)^2));
        W(y_t,:) = W(y_t,:) + eta_t*x_t;
        W(s_t,:) = W(s_t,:) - eta_t*x_t;
    end
    
    
    run_time=toc;
    if (mod(t,t_tick)==0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        TMs=[TMs run_time];
    end
end
classifier.W = W;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
