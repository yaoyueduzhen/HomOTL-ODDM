function [classifier, err_count, run_time, mistakes, mistakes_idx] = HomOTL_ODDM(Y, A, X, MEAN_Xs,MEAN_Xt,NUM_t,MEAN_ks,MEAN_kt,NUM_kt, options, id_list,classifiers)

%% initialize parameters
beta = options.beta1;
C = options.C; 
T_TICK = options.t_tick;
k = options.k;
mu = options.mu;

numSources = length(classifiers);

u_t = [];
v_t = [];
for i = 1:numSources
     Ws{i} = classifiers(i).W;
     
     Wt{i} = zeros(k,options.dim);
     
     u_t = [u_t, 1/(2*numSources)];
     v_t = [v_t, 1/(2*numSources)];
     
end

for i = 1:numSources,
    p_s{i} = u_t(i) / (sum(u_t, 2) + sum(v_t,2));
    p_t{i} = v_t(i) / (sum(u_t, 2) + sum(v_t,2));
end

ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [0];

t_tick = T_TICK; 
%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    
    for i = 1:numSources
        id_new = id;
        x_t = X{i}(id_new, :);
        y_t = Y{i}(id_new);
        x_t = x_t*A{i};
        x_t = x_t*(1/sqrt(sum(x_t.^2,2)));
        F_s{i} = Ws{i}*x_t'; 
        F_t{i} = Wt{i}*x_t';
    end
    
    for i = 1:numSources
        p_s{i} = u_t(i) / (sum(u_t, 2) + sum(v_t,2));
        p_t{i} = v_t(i) / (sum(u_t, 2) + sum(v_t,2));
    end
    
    for i = 1:numSources
        u_t(i) = p_s{i};
        v_t(i) = p_t{i};
    end
    
    F = 0;
    for i = 1:numSources
        F = F + p_s{i}*F_s{i} + p_t{i}*F_t{i};
    end

    [F_max,hat_y_t]=max(F);

    
    % count accumulative mistakes
    if (hat_y_t~=y_t),
        err_count = err_count + 1;
    end
    
    for i = 1:numSources
        [F_max1,hat_y_t1]=max(F_s{i});
        [F_max2,hat_y_t2]=max(F_t{i});
        z_1 = (hat_y_t1~=y_t);
        z_2 = (hat_y_t2~=y_t);
        u_t(i)=u_t(i)*beta^z_1;
        v_t(i)=v_t(i)*beta^z_2;
    end
    
    for i = 1:numSources
        id_new = id;
        x_t = X{i}(id_new, :);
        y_t = Y{i}(id_new);
        x_t = x_t*A{i};
        x_t = x_t*(1/sqrt(sum(x_t.^2,2)));
        
        
        Fs2=F_t{i};
        Fs2(y_t)=-inf;
        [Fs_max2, s_t2]=max(Fs2);
        l_t2 = max(0, 1 - (F_t{i}(y_t) - F_t{i}(s_t2))); 

        if (l_t2 > 0),
            eta_t = min(C, l_t2/(2*norm(x_t)^2));
            Wt{i}(y_t,:) = Wt{i}(y_t,:) + eta_t*x_t;
            Wt{i}(s_t2,:) = Wt{i}(s_t2,:) - eta_t*x_t;
        end
    end
    
    for i = 1:numSources
        id_new = id;
        x_t = X{i}(id_new, :);
        y_t = Y{i}(id_new);
        
        MEAN_Xt{i} = (MEAN_Xt{i} .* NUM_t{i} + x_t)./(NUM_t{i} + 1);
        NUM_t{i} = NUM_t{i} + 1;
        
        MEAN_kt{i}(y_t,:) =(MEAN_kt{i}(y_t,:).*NUM_kt{i}(y_t) + x_t)./(NUM_kt{i}(y_t) + 1);
        NUM_kt{i}(y_t) = NUM_kt{i}(y_t) + 1;
        
        X_t{i} =  MEAN_Xs{i} - MEAN_Xt{i};
        X_kt{i} = MEAN_ks{i} - MEAN_kt{i};
        
        if mod(t,10)==0
            B = eye(size(x_t,2),size(x_t,2)) + mu*X_t{i}'*X_t{i};
            for j=1:k
                B = B + mu*X_kt{i}(j,:)'*X_kt{i}(j,:);
            end
            if(det(B)~=0)
                A{i} = inv(B)*A{i};
            else
                A{i} = pinv(B)*A{i};
            end
        end
    end
   
    run_time=toc;
    if t<T_TICK  
        if (t==t_tick)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
            
            t_tick=2*t_tick;
            if t_tick>=T_TICK,
                t_tick = T_TICK;
            end
            
        end
    else
        if (mod(t,t_tick)==0)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
            
        end
    end
end
classifier.Ws = Ws;
classifier.Wt = Wt;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
