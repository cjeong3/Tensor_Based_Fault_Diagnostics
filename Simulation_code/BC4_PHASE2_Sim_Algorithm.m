%% SIMULATION STUDY

% #########################################################################
% Title: (PHASE2) Benchmark IV - Two-dimensional variable selection
% Output Path: /Simulation_result/Benchmark/
% #########################################################################

clc; clear;
addpath('./tensor_toolbox-v3.1')

%% Setting
ImportData='DataTucker2'; % imported data: DataCP1 DataCP2 DataTucker1 DataTucker2
NN=200; % number of data samples: 100 300 500

load(['./Simulation_data/' ImportData '_' num2str(NN) '.mat'])

Input=XA; % input data XA:IID, XB:Correlated
DataName='XA';

nTry=30; % number of experiments
lambdas=(0:0.5:10); % tuning parameters
ranks=(1:1:4); % rank
epsilon=0.001; % termination tolerance  
rho = 0.5; % decay factor in linesearch
tau0 = 0.001; % initial step size

noiselevel=1; % noise level 1-5 (0.2-1.0)

%% Algorithm
rng('default')
Idx1org=sum(beta.U{1},2)==0;
Idx2org=sum(beta.U{2},2)==0;
Idx2org=Idx2org';

nLambda=length(lambdas);
nRank=length(ranks);

rslt=[];

tic;
for iTry = 1:nTry 
    iTry
    
    ProcessTen = Input{iTry};   
    Noise = noisetensor{iTry};
    
    Xtrue=ProcessTen(1:NN,:); % true process tensor 
    
    % noise level setting
    if noiselevel == 1 
        for i=1:NN
            temp = tensor(double(Xtrue{i}) + double(Noise{i})*0.2);
            X{i,1} = temp;
        end
    elseif noiselevel == 2 
        for i=1:NN
            temp = tensor(double(Xtrue{i}) + double(Noise{i})*0.4);
            X{i,1} = temp;
        end    
    elseif noiselevel == 3
        for i=1:NN
            temp = tensor(double(Xtrue{i}) + double(Noise{i})*0.6);
            X{i,1} = temp;
        end
    elseif noiselevel == 4 
        for i=1:NN
            temp = tensor(double(Xtrue{i}) + double(Noise{i})*0.8);
            X{i,1} = temp;
        end
    elseif noiselevel == 5 
        for i=1:NN
            temp = tensor(double(Xtrue{i}) + double(Noise{i})*1);
            X{i,1} = temp;
        end    
    end
        
    % true (X,Y) pair
    vecX=[]; Ytmp=[];   
    for i=1:NN
        vecX{i}=reshape(double(Xtrue{i}),1,p1*p2*p3);
        Ytmp = [Ytmp 1/(1+exp(0-(vecX{i}*vecBeta')))]; 
    end
    Ybin=(Ytmp>0.5)';
    Y=Ybin(1:NN); 
    
    % make from tensor to matrix 
    for i=1:length(X)
        X{i} = mean(double(X{i}),3);
    end
  
    BetaCell={}; Beta0Cell={};
    
    % initialization 
    Xele=cell(p1,p2); 
    Btilde=zeros(p1,p2);
    for p=1:p1
        for q=1:p2
            for m=1:NN        
                Xele{p,q}=[Xele{p,q}; X{m}(p,q)];
            end
            b1=glmfit(Xele{p,q},Y,'binomial','link','logit','Constant','off'); 
            Btilde(p,q)=b1;            
        end
    end    
    clear Xele;

    [C,S,D] = svd(Btilde); 

    for j = 1:nRank
        U0arr{j,1} = C(:,1:j);
        V0arr{j,1} = D(1:j,:);
    end       
    betaIN=U0arr{j,1}*V0arr{j,1}; 
    
    for i=1:nLambda % lambda
        for j=1:nRank % rank
            
            fprintf('iTry: %d \n',iTry) 
            fprintf('rank: %d \n',ranks(j))
            fprintf('lambda: %d (%d) \n',i,lambdas(i))
            fprintf('#sample: %d \n',NN)
            
            beta0=betaIN;
            U=U0arr{j,1}; 
            V=V0arr{j,1}; 
            Iter=1; negloglik_penalty_new_record=[];        
            
            b = 0;           
            while 1
                % 1-1.update U 
                gradUnll = 0;
                for m = 1:NN
                    Xmat = X{m};
                    Xi = Xmat*V';
                    temp = - ( Y(m) - exp(b+U(:)'*Xi(:))/(1+exp(b+U(:)'*Xi(:))) )*Xi;
                    gradUnll = gradUnll + temp;
                end
                gamma = sqrt(ranks(j));
                lambda = lambdas(i);                
                
                % 1-2.constant step size 
                tau = tau0;                 
                
%                 % 1-2.linesearch
%                 tau = tau0; 
%                 gradUnll_prox = ( U - bc_prox_U(U-tau*gradUnll,lambda,gamma,tau) )/tau;
%                 Unew = U - tau*gradUnll_prox;
%                 while bc_nll_fun_U(Unew,V,b,X,Y,NN) > bc_nll_fun_U(U,V,b,X,Y,NN) - tau*gradUnll(:)'*gradUnll_prox(:) + 0.5*tau*norm(gradUnll_prox,2)^2
%                     tau = tau*rho;
%                     gradUnll_prox = ( U - bc_prox_U(U-tau*gradUnll,lambda,gamma,tau) )/tau;
%                     Unew = U - tau*gradUnll_prox;
%                 end 

                U = bc_prox_U(U-tau*gradUnll,lambda,gamma,tau);
                
                % 2-1.update V
                gradVnll = 0;
                for m = 1:NN
                    Xmat = X{m};
                    Xi = U'*Xmat;
                    temp = - ( Y(m) - exp(b+V(:)'*Xi(:))/(1+exp(b+V(:)'*Xi(:))) )*Xi;
                    gradVnll = gradVnll + temp;
                end
                
                % 2-2.constant step size 
                tau = tau0;  
                
                % 2-2.linesearch               
%                 tau = tau0;
%                 gradVnll_prox = ( V - bc_prox_V(V-tau*gradVnll,lambda,gamma,tau) )/tau;
%                 Vnew = V - tau*gradVnll_prox;
%                 while bc_nll_fun_V(U,Vnew,b,X,Y,NN) > bc_nll_fun_V(U,V,b,X,Y,NN) - tau*gradVnll(:)'*gradVnll_prox(:) + 0.5*tau*norm(gradVnll_prox,2)^2
%                     tau = tau*rho;
%                     gradVnll_prox = ( V - bc_prox_V(V-tau*gradVnll,lambda,gamma,tau) )/tau;
%                     Vnew = V - tau*gradVnll_prox;
%                 end 

                V = bc_prox_V(V-tau*gradVnll,lambda,gamma,tau); 
                
                % 3-1.update b = alpha 
                gradbnll = 0;
                for m = 1:NN
                    Xmat = X{m};
                    Xi = Xmat*V';
                    temp = - ( Y(m) - exp(b+U(:)'*Xi(:))/(1+exp(b+U(:)'*Xi(:))) );
                    gradbnll = gradbnll + temp;
                end
                
                % 3-2.constant step size 
                tau = tau0;                  
                
%                 % 3-2.linesearch                  
%                 tau = tau0;      
%                 while bc_nll_fun_b(U,V,b-tau*gradbnll,X,Y,NN) > bc_nll_fun_b(U,V,b,X,Y,NN) - 0.5*tau*norm(gradbnll,2)^2
%                     tau = tau*rho;
%                 end 

                b = b-tau*gradbnll;
                
                % objective and neg.log-likelihood
                negloglik = bc_nll_fun_U(U,V,b,X,Y,NN); 
                negloglik_penalty = negloglik+lambdas(i)*( sum(sqrt(ranks(j))*sqrt(sum(U.*U,2)))+sum(sqrt(ranks(j))*sqrt(sum(V.*V,1))) );

                % termination criteria 
                negloglik_penalty_new=negloglik_penalty;
                U_new=U; V_new=V; 
                if Iter~=1
                    delta=abs(negloglik_penalty_old-negloglik_penalty_new);
                    if (delta<epsilon) || (Iter>500) 
                        break;
                    end
                end
                negloglik_penalty_old=negloglik_penalty_new;
                U_old=U_new; V_old=V_new; 
                Iter=Iter+1;
                negloglik_penalty_new_record=[negloglik_penalty_new_record negloglik_penalty_new];    
            end
            delta
            Iter
            
            U(abs(U)<10e-5)=0
            V(abs(V)<10e-5)=0
            b
            
            Index1=sum(U,2)==0;
            Index2=sum(V,1)==0;
                
            % unpenalized regression 
            XNew = X;
            XXX = [];
            for m = 1:NN
                XX = XNew{m};
                XX(Index1,:) = [];
                XX(:,Index2) = [];
                XXX = [XXX;XX(:)'];
            end
            mdl = fitglm(XXX,Y,'Distribution','binomial');
            
            % model selection criterion:AICc
            AIC2(i,j)=mdl.ModelCriterion.AICc; 
            BIC2(i,j)=mdl.ModelCriterion.CAIC; 
            
            B=U*V;
            BetaCell{i,j}=B;
            UCell{i,j}=U;
            VCell{i,j}=V;
            Beta0Cell{i,j}=beta0;           
        end      
    end 

    %% condition for small sample size <= 200
    AIC2(AIC2<=-1000) = inf;

    [M,I] = min(AIC2(:));
    [f,g] = ind2sub(size(AIC2),I);    

    Bout = BetaCell{f,g}; 
    Uout = UCell{f,g};
    Vout = VCell{f,g};
    
    Idx1out=sum(Uout,2)==0;
    Idx2out=sum(Vout,1)==0;
    
    TN=sum((Idx1org==Idx1out).*Idx1org) + sum((Idx2org==Idx2out).*Idx2org); 
    TP=sum((sum(Uout,2)~=0).*(~Idx1org)) + sum((sum(Vout,1)~=0).*(~Idx2org)); 
    Yzero=sum(Idx1org) + sum(Idx2org);
    Nzero=sum(~Idx1org) + sum(~Idx2org);
    FN=Nzero-TP;
    FP=Yzero-TN;

    Acrcy=100*(TP+TN)/(Yzero+Nzero);
     
    iReport=[TP TN FN FP Acrcy];
    rslt=[rslt; iReport];

    %% Save results 
    % save(['./Simulation_result/BC/Record/Result_BC4_' ImportData '_' DataName num2str(NN) '_noise' num2str(noiselevel) '_iTry' num2str(iTry) '.mat'])
end 
elapsed_time = toc;

%% Save results 
save(['./Simulation_result/Benchmark/Result_BC4_' ImportData '_' DataName num2str(NN) '_noise' num2str(noiselevel) '.mat'])

