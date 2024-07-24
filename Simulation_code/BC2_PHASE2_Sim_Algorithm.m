%% SIMULATION STUDY

% #########################################################################
% Title: (PHASE2) Benchmark II - Process variables -> Stages
% Output Path: /Simulation_result/Benchmark/
% #########################################################################

clc; clear;
addpath('./tensor_toolbox-v3.1')

%% Setting
ImportData='DataCP1'; % imported data: DataCP1 DataCP2 DataTucker1 DataTucker2
NN=200; % number of data samples: 100 300 500

load(['./Simulation_data/' ImportData '_' num2str(NN) '.mat'])

Input=XA; % input data XA:IID, XB:Correlated
DataName='XA';

nTry=30; % number of experiments
lambdas=(0:2:26); % tuning parameters
epsilon=0.001; % termination tolerance
rho = 0.5; % decay factor in linesearch
tau0 = 0.001; % initial step size

noiselevel=1; % noise level 1-5 (0.2-1.0)

%% Algorithm
rng('default')
Idx1org=sum(beta.U{1},2)==0;
Idx2org=sum(beta.U{2},2)==0;

nLambda=length(lambdas);

rslt=[];
AICc_rslt = [];

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
    
    BetaCell={}; Beta0Cell={};
    
    % Initialization 
    Xele=cell(p1,p2,p3); 
    betaTilde=zeros(p1,p2,p3);
    for p=1:p1
        for q=1:p2
            for l=1:p3
                for m=1:NN        
                    Xele{p,q,l}=[Xele{p,q,l}; X{m}(p,q,l)];
                end
                b1=glmfit(Xele{p,q,l},Y,'binomial','link','logit','Constant','off'); 
                betaTilde(p,q,l)=b1;            
            end
        end
    end    
    clear Xele;

    %=========================
    % Step1: process variable
    %=========================
    
    for i=1:nLambda % lambda

        fprintf('iTry: %d \n',iTry) 
        fprintf('Step1: process variable selection \n')
        fprintf('lambda: %d (%d) \n',i,lambdas(i))
        fprintf('#sample: %d \n',NN)

        % initialize coefficient tensor
        betaIN=betaTilde;

        beta0=betaIN;
        B = beta0;
        
        Iter=1; negloglik_penalty_new_record=[];

        b = 0;
        while 1
            % 1-1.update horizontal slices (process variable mode)  
            gradBnll = 0;
            for m = 1:NN
                Xmat = double(X{m});
                temp = - ( Y(m) - exp(b+B(:)'*Xmat(:))/(1+exp(b+B(:)'*Xmat(:))) )*Xmat;
                gradBnll = gradBnll + temp;
            end                      
            gamma = sqrt(p2*p3);
            lambda = lambdas(i);
            
            % 1-2.constant step size  
            tau = tau0; 
            
            % 1-2.linesearch 
%             tau = tau0;
%             gradBnll_prox = ( B - bc2_prox_B(B-tau*gradBnll,lambda,gamma,tau) )/tau;
%             Bnew = B - tau*gradBnll_prox;
%             while bc2_nll_fun_B(Bnew,b,X,Y,NN) > bc2_nll_fun_B(B,b,X,Y,NN) - tau*gradBnll(:)'*gradBnll_prox(:) + 0.5*tau*norm(gradBnll_prox,2)^2
%                 tau = tau*rho;
%                 gradBnll_prox = ( B - bc2_prox_B(B-tau*gradBnll,lambda,gamma,tau) )/tau;
%                 Bnew = B - tau*gradBnll_prox;
%             end

            B = bc2_prox_B(B-tau*gradBnll,lambda,gamma,tau);

            % 2-1.update b = alpha 
            gradbnll = 0;
            for m = 1:NN
                Xmat = double(X{m});
                temp = - ( Y(m) - exp(b+B(:)'*Xmat(:))/(1+exp(b+B(:)'*Xmat(:))) );
                gradbnll = gradbnll + temp;
            end
            
            % 2-1.constant step size  
            tau = tau0; 
            
%             % 2-1.linesearch                  
%             tau = tau0;    
%             while bc2_nll_fun_b(B,b-tau*gradbnll,X,Y,NN) > bc2_nll_fun_b(B,b,X,Y,NN) - 0.5*tau*norm(gradbnll,2)^2
%                 tau = tau*rho;
%             end 

            b = b-tau*gradbnll;

            % objective and neg.log-likelihood
            negloglik = bc2_nll_fun_B(B,b,X,Y,NN);               
            negloglik_penalty = negloglik+lambdas(i)*( sum( sqrt(gamma)*sqrt(sum(sum(B.^2,3),2)) ) );             
            
            % termination criteria
            negloglik_penalty_new=negloglik_penalty
            B_new=B;
            if Iter~=1   
                delta=abs(negloglik_penalty_old-negloglik_penalty_new); 
                zeta=sum(sum(sum((B_old-B_new).^2)));
                if (delta<epsilon) || (Iter>500) 
                    break;
                end
            end
            negloglik_penalty_old=negloglik_penalty_new;
            B_old=B_new;
            Iter=Iter+1;
            negloglik_penalty_new_record=[negloglik_penalty_new_record negloglik_penalty_new];                
        end
        delta
        Iter

        B(abs(B)<10e-5)=0
        betaIN=B

        Index1=sum(sum(B,3),2)==0;

        % unpenalized regression 
        XNew = X;
        XXX = [];
        for m = 1:NN
            XX = double(XNew{m});
            XX(Index1,:,:) = [];
            % XX(:,Index2,:) = [];
            XXX = [XXX;XX(:)'];
        end
        mdl = fitglm(XXX,Y,'Distribution','binomial');
        LLE(i,1) = mdl.LogLikelihood;        
        
        % model selection criterion:AICc
        AIC2(i,1)=mdl.ModelCriterion.AICc; 
        BIC2(i,1)=mdl.ModelCriterion.CAIC; 

        BetaCell{i,1}=betaIN;
        Beta0Cell{i,1}=beta0;           
    end      

    [M,I] = min(AIC2);
    [f,g] = ind2sub(size(AIC2),I);
    Bout = BetaCell{f,1}; % check 
    
    % nonzero process variable mode index 
    nzeroidx = sum(sum(Bout,3),2) ~=0;
   
    % remove sparse process variable mode
    Xn = {};
    for m = 1:NN
        XX = double(X{m});
        Xn{m,1} = XX(nzeroidx,:,:);
        % XX(:,Index2,:) = [];
    end

    % number of rows 
    w = size(Xn{1},1);  
    
    BetaCelln={}; Beta0Celln={};
    
    % initialization 
    Xelen=cell(w,p2,p3); 
    betaTilden=zeros(w,p2,p3);
    for p=1:w
        for q=1:p2
            for l=1:p3
                for m=1:NN        
                    Xelen{p,q,l}=[Xelen{p,q,l}; Xn{m}(p,q,l)];
                end
                b1=glmfit(Xelen{p,q,l},Y,'binomial','link','logit','Constant','off'); 
                betaTilden(p,q,l)=b1;            
            end
        end
    end    
    clear Xelenewn;
    
    %=========================
    % Step2: stage
    %=========================  
    
    for i=1:nLambda % lambda

        fprintf('iTry: %d \n',iTry) 
        fprintf('Step2: stage \n')
        fprintf('lambda: %d (%d) \n',i,lambdas(i))
        fprintf('#sample: %d \n',NN)

        % initialize coefficient tensor
        betaIN=betaTilden;

        beta0=betaIN;
        Bn = beta0;
        
        Iter=1; negloglik_penalty_new_record=[];

        b = 0;
        while 1
            % 1-1.update lateral slices (stage mode)  
            gradBnll = 0;
            for m = 1:NN
                Xmat = double(Xn{m});
                temp = - ( Y(m) - exp(b+Bn(:)'*Xmat(:))/(1+exp(b+Bn(:)'*Xmat(:))) )*Xmat;
                gradBnll = gradBnll + temp;
            end
            gamma = sqrt(w*p3);
            lambda = lambdas(i);            
            
            % 1-2.constant step size
            tau = tau0; 
            
%             % 1-2.linesearch         
%             tau = tau0; 
%                 gradBnll_prox = ( Bn - bc3_prox_B(Bn-tau*gradBnll,lambda,gamma,tau) )/tau;
%                 Bnew = Bn - tau*gradBnll_prox;
%                 while bc3_nll_fun_B(Bnew,b,Xn,Y,NN) > bc3_nll_fun_B(B,b,Xn,Y,NN) - tau*gradBnll(:)'*gradBnll_prox(:) + 0.5*tau*norm(gradBnll_prox,2)^2
%                     tau = tau*rho;
%                     gradBnll_prox = ( Bn - bc3_prox_B(Bn-tau*gradBnll,lambda,gamma,tau) )/tau;
%                     Bnew = Bn - tau*gradBnll_prox;
%                 end 

            Bn = bc3_prox_B(Bn-tau*gradBnll,lambda,gamma,tau);

            % 2-1.update b = alpha 
            gradbnll = 0;
            for m = 1:NN
                Xmat = double(Xn{m});
                temp = - ( Y(m) - exp(b+Bn(:)'*Xmat(:))/(1+exp(b+Bn(:)'*Xmat(:))) );
                gradbnll = gradbnll + temp;
            end
            
            % 2-2.constant step size 
            tau = tau0;             
            
            % 2-2.Linesearch                  
%             tau = tau0;    
%             while bc2_nll_fun_b(B,b-tau*gradbnll,X,Y,NN) > bc2_nll_fun_b(B,b,X,Y,NN) - 0.5*tau*norm(gradbnll,2)^2
%                 tau = tau*rho;
%             end 

            b = b-tau*gradbnll;

            % objective and neg.log-likelihood
            negloglik = bc3_nll_fun_B(Bn,b,Xn,Y,NN);               
            negloglik_penalty = negloglik+lambdas(i)*( sum( sqrt(gamma)*sqrt(sum(sum(Bn.^2,1),3)) ) );             
            
            % termination criteria 
            negloglik_penalty_new=negloglik_penalty
            Bn_new=Bn;
            if Iter~=1   
                delta=abs(negloglik_penalty_old-negloglik_penalty_new);
                zeta=sum(sum(sum((Bn_old-Bn_new).^2)));
                if (delta<epsilon) || (Iter>500) 
                    break;
                end
            end
            negloglik_penalty_old=negloglik_penalty_new;
            Bn_old=Bn_new;
            Iter=Iter+1;
            negloglik_penalty_new_record=[negloglik_penalty_new_record negloglik_penalty_new];                
        end
        delta
        Iter

        Bn(abs(Bn)<10e-5)=0
        betaIN=Bn

        Index2=sum(sum(Bn,3),1)==0;

        % unpenalized regression 
        XNew = Xn;
        XXX = [];
        for m = 1:NN
            XX = double(XNew{m});
%             XX(Index1,:,:) = [];
            XX(:,Index2,:) = [];
            XXX = [XXX;XX(:)'];
        end
        mdl = fitglm(XXX,Y,'Distribution','binomial');
        LLE(i,1) = mdl.LogLikelihood;        
        
        % model selection criterion:AICc
        AIC2(i,1)=mdl.ModelCriterion.AICc; 
        BIC2(i,1)=mdl.ModelCriterion.CAIC; 

        BetaCelln{i,1}=betaIN;
        Beta0Celln{i,1}=beta0;           
    end      

    %% condition for small sample size <= 200
    AIC2(AIC2<=-1000) = inf;

    [M,I] = min(AIC2);
    [f,g] = ind2sub(size(AIC2),I);
    Boutn = BetaCelln{f,1}; % check     
       
    Idx1out=sum(sum(Bout,3),2)==0;
    Idx2out=sum(sum(Boutn,3),1)==0;
    Idx2out=Idx2out';
    
    TN=sum((Idx1org==Idx1out).*Idx1org) + sum((Idx2org==Idx2out).*Idx2org); 
    TP=sum((sum(sum(Bout,3),2)~=0).*(sum(beta.U{1},2)~=0)) + sum((sum(sum(Boutn,3),1)~=0)'.*(sum(beta.U{2},2)~=0)); 
    Yzero=sum(sum(beta.U{1},2)==0) + sum(sum(beta.U{2},2)==0);
    Nzero=sum(sum(beta.U{1},2)~=0) + sum(sum(beta.U{2},2)~=0);
    FN=Nzero-TP;
    FP=Yzero-TN;
        
    Acrcy=100*(TP+TN)/(Yzero+Nzero);
    
    iReport=[TP TN FN FP Acrcy];
    rslt=[rslt; iReport]; 

    %% Save results 
    % save(['./Simulation_result/Benchmark/Record/Result_BC2_' ImportData '_' DataName num2str(NN) '_noise' num2str(noiselevel) '_iTry' num2str(iTry) '.mat'])
end 
elapsed_time = toc;

%% Save results 
save(['./Simulation_result/Benchmark/Result_BC2_' ImportData '_' DataName num2str(NN) '_noise' num2str(noiselevel) '.mat'])

