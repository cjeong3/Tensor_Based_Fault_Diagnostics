% SIMULATION STUDY

% #########################################################################
% Title: (PHASE2) Benchmark I - Tucker-based method with rank (1,1,1)
% Output Path: /Simulation_result/Benchmark/
% #########################################################################

clc; clear;
addpath('./tensor_toolbox-v3.1')

%% Setting
ImportData='DataTucker1'; % imported data: DataTucker1 DataTucker2
NN=200; % number of data samples

load(['./Simulation_data/' ImportData '_' num2str(NN) '.mat'])

Input=XA; % input data XA:IID, XB:Correlated
DataName='XA';

nTry=30; % number of experiment 
lambdas=(0:2:26); % tuning parameters
ranks=(1:1:31); % rank
epsilon=0.001; % termination tolerance  
M=3; % upto what rank tested
rho=0.5; % decay factor in linesearch
tau0=0.001; % initial stepsize
rr=1; % test rank

noiselevel=1; % noise level 1-5 (0.2-1.0)

%% Algorithm
rng('default')
Idx1org=sum(beta.U{1},2)==0;
Idx2org=sum(beta.U{2},2)==0;

nLambda=length(lambdas);
nRank=length(ranks);

% clear C;
% [C{M:-1:1}] = ndgrid(1:3); % ndgrid(1:2);
% C=reshape(cat(M+1,C{:}),[],M);
% rankvec={};
% for i=1:length(C)
%     rankvec=[rankvec; C(i,:)];
% end
% rankvec = {[1,1,2], [2,1,2], [3,1,2], [4,1,2], [5,1,2], [6,1,2], [7,1,2], ...
%            [2,1,2], [2,2,2], [2,3,2], [2,4,2], [2,5,2], [2,6,2], [2,7,2], ...
%            [2,1,1], [2,1,2], [2,1,3], [2,1,4], [2,1,5], [2,1,6], [2,1,7], ...
%            [2,8,2], [1,1,1], [3,2,2], [4,3,2], [5,4,2], [6,5,2], [7,6,2], ...
%            [3,2,3], [4,3,4] ,[5,4,5]}';    
rankvec = {[1,1,1]}';   
    
tmp=[];
for a=1:nRank; tmp=[tmp repmat(a,1,nLambda)]; end
indexV = [tmp' repmat(1:nRank,1,nLambda)'];

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
    negloglik_penalty_cell={};
    G_record_cell={};
    U_record_cell={};
    V_record_cell={};
    W_record_cell={};
    b_record_cell={};

    % initialization 
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
        
    for j=rr % rank
        for i=1:nLambda % lambda          
            
            fprintf('iTry: %d \n',iTry) 
            fprintf('rank: %s \n',num2str(rankvec{j}))
            fprintf('lambda: %d (%d) \n',i,lambdas(i))
            fprintf('#sample: %d \n',NN)
            
            % initialize basis/factor matrices 
            betaIN=tucker_als(tensor(betaTilde),rankvec{j});

            beta0=betaIN;
            G=betaIN.core;
            U=betaIN.U{1}; % U = B1
            V=betaIN.U{2}; % V = B2
            W=betaIN.U{3}; % W = B3
            Iter=1; 
            negloglik_penalty_new_record=[];
            G_record={};
            U_record={};
            V_record={};
            W_record={};
            b_record=[];
            rank = rankvec{j};
            
            b = 0;
            while 1
                % 1-1.update G 
                lambda = lambdas(i);
                Gtenmat = tenmat(G,1);
                Gmat = Gtenmat.data;
                gradGnll = 0;
                for m = 1:NN
                    Xtenmat = tenmat(X{m},1);
                    Xmat = Xtenmat.data;
                    Xi = U'*Xmat*kron(W,V);
                    temp = - ( Y(m) - exp(b+Gmat(:)'*Xi(:))/(1+exp(b+Gmat(:)'*Xi(:))) )*Xi;
                    gradGnll = gradGnll + temp;
                end
                
                 % 1-2.constant step size 
                tau = tau0;                
                
%                 % 1-2.linesearch
%                 tau = tau0; 
%                 while tucker_nll_fun_G(Gmat-tau*gradGnll,U,V,W,b,X,Y,NN) > tucker_nll_fun_G(Gmat,U,V,W,b,X,Y,NN) - 0.5*tau*norm(gradGnll,2)^2
%                     tau = tau*rho;
%                 end 
%                 Gmat = Gmat-tau*gradGnll;

                Gmat = tucker_prox_G(Gmat-tau*gradGnll,lambda,tau);
                G = tensor(Gmat(:),size(G));

                % 2-1.update U = B1
                Gtenmat = tenmat(G,1);
                Gmat = Gtenmat.data;
                gradUnll = 0;
                for m = 1:NN
                    Xtenmat = tenmat(X{m},1);
                    Xmat = Xtenmat.data;
                    Xi = Xmat*kron(W,V)*Gmat';
                    temp = - ( Y(m) - exp(b+U(:)'*Xi(:))/(1+exp(b+U(:)'*Xi(:))) )*Xi;
                    gradUnll = gradUnll + temp;
                end
                gamma = sqrt(rank(1));
                lambda = lambdas(i);
                
                % 2-2.constant step size 
                tau = tau0;                   
                
%                 % 2-2.linesearch
%                 gradUnll_prox = ( U - tucker_prox_U(U-tau*gradUnll,lambda,gamma,tau) )/tau;
%                 Unew = U - tau*gradUnll_prox;
%                 while tucker_nll_fun_U(Gmat,Unew,V,W,b,X,Y,NN) > tucker_nll_fun_U(Gmat,U,V,W,b,X,Y,NN) - tau*gradUnll(:)'*gradUnll_prox(:) + 0.5*tau*norm(gradUnll_prox,2)^2
%                     tau = tau*rho;
%                     gradUnll_prox = ( U - tucker_prox_U(U-tau*gradUnll,lambda,gamma,tau) )/tau;
%                     Unew = U - tau*gradUnll_prox;
%                 end 

                U = tucker_prox_U(U-tau*gradUnll,lambda,gamma,tau); 

                % 3-1.Update V = B2
                Gtenmat = tenmat(G,2);
                Gmat = Gtenmat.data;
                gradVnll = 0;
                for m = 1:NN
                    Xtenmat = tenmat(X{m},2);
                    Xmat = Xtenmat.data;
                    Xi = Xmat*kron(W,U)*Gmat';
                    temp = - ( Y(m) - exp(b+V(:)'*Xi(:))/(1+exp(b+V(:)'*Xi(:))) )*Xi;
                    gradVnll = gradVnll + temp;
                end
                gamma = sqrt(rank(2));
                lambda = lambdas(i);
                
                % 3-2.constant step size 
                tau = tau0;                 
                
%                 % 3-2.linesearch               
%                 tau =  tau0;
%                 gradVnll_prox = ( V - tucker_prox_V(V-tau*gradVnll,lambda,gamma,tau) )/tau;
%                 Vnew = V - tau*gradVnll_prox;
%                 while tucker_nll_fun_V(Gmat,U,Vnew,W,b,X,Y,NN) > tucker_nll_fun_V(Gmat,U,V,W,b,X,Y,NN) - tau*gradVnll(:)'*gradVnll_prox(:) + 0.5*tau*norm(gradVnll_prox,2)^2
%                     tau = tau*rho;
%                     gradVnll_prox = ( V - tucker_prox_V(V-tau*gradVnll,lambda,gamma,tau) )/tau;
%                     Vnew = V - tau*gradVnll_prox;
%                 end 

                V = tucker_prox_V(V-tau*gradVnll,lambda,gamma,tau); 

                % 4-1.update W = B3
                Gtenmat = tenmat(G,3);
                Gmat = Gtenmat.data;
                gradWnll = 0;
                for m = 1:NN
                    Xtenmat = tenmat(X{m},3);
                    Xmat = Xtenmat.data;
                    Xi = Xmat*kron(V,U)*Gmat';
                    temp = - ( Y(m) - exp(b+W(:)'*Xi(:))/(1+exp(b+W(:)'*Xi(:))) )*Xi;
                    gradWnll = gradWnll + temp;
                end

                % 4-2.constant step size 
                tau = tau0;                 
                
%                 % 4-2.linesearch               
%                 tau = tau0; 
%                 while tucker_nll_fun_W(Gmat,U,V,W-tau*gradWnll,b,X,Y,NN) > tucker_nll_fun_W(Gmat,U,V,W,b,X,Y,NN) - 0.5*tau*norm(gradWnll,2)^2
%                     tau = tau*rho;
%                 end 
%                 W = W-tau*gradWnll;    

                W = tucker_prox_W(W-tau*gradWnll,lambda,tau);

                % 5-1.update b = alpha 
                Gtenmat = tenmat(G,1);
                Gmat = Gtenmat.data;
                gradbnll = 0;
                for m = 1:NN
                    Xtenmat = tenmat(X{m},1);
                    Xmat = Xtenmat.data;
                    Xi = U'*Xmat*kron(W,V);
                    temp = - ( Y(m) - exp(b+Gmat(:)'*Xi(:))/(1+exp(b+Gmat(:)'*Xi(:))) );
                    gradbnll = gradbnll + temp;
                end
                
                % 5-2.constant step size 
                tau = tau0;      
                
%                 % 5-2.linesearch                  
%                 tau = tau0;      
%                 while tucker_nll_fun_b(Gmat,U,V,W,b-tau*gradbnll,X,Y,NN) > tucker_nll_fun_b(Gmat,U,V,W,b,X,Y,NN) - 0.5*tau*norm(gradbnll,2)^2
%                     tau = tau*rho;
%                 end 

                b = b-tau*gradbnll;
                G = tensor(Gmat(:),size(G));     

                % objective and neg.log-likelihood
                negloglik = tucker_nll_fun_U(Gmat,U,V,W,b,X,Y,NN);               
                negloglik_penalty = negloglik+lambdas(i)*( sum(sqrt(rank(1))*sqrt(sum(U.*U,2)))+sum(sqrt(rank(2))*sqrt(sum(V.*V,2))) );             
                
                % termination criteria 
                negloglik_penalty_new=negloglik_penalty
                G_new=G; U_new=U; V_new=V; W_new=W;
                if Iter~=1
                    delta=abs(negloglik_penalty_old-negloglik_penalty_new);
                    if (delta<epsilon) || (Iter>500) 
                        break;
                    end
                end
                negloglik_penalty_old=negloglik_penalty_new;
                G_old=G_new; U_old=U_new; V_old=V_new; W_old=W_new;
                negloglik_penalty_new_record=[negloglik_penalty_new_record negloglik_penalty_new];              
                G_record{Iter,1} = G_new;
                U_record{Iter,1} = U_new;
                V_record{Iter,1} = V_new;
                W_record{Iter,1} = W_new;
                b_record=[b_record b];
                Iter=Iter+1;                
            end
            delta
            Iter
            
            Gmat
            U(abs(U)<10e-5)=0
            V(abs(V)<10e-5)=0
            W
            betaIN=ttensor(G,{U,V,W});
            b
            
            Index1=sum(U,2)==0;
            Index2=sum(V,2)==0;
            
            % unpenalized regression 
            XNew = X;
            XXX = [];
            for m = 1:NN
                XX = double(tensor(XNew{m}));
                XX(Index1,:,:) = [];
                XX(:,Index2,:) = [];
                XXX = [XXX;XX(:)'];
            end
            mdl = fitglm(XXX,Y,'Distribution','binomial');
            
            % model selection criterion:AICc
            AIC2(i,j)=mdl.ModelCriterion.AICc; 
            BIC2(i,j)=mdl.ModelCriterion.CAIC; 
            
            BetaCell{i,j}=betaIN;
            Beta0Cell{i,j}=beta0;      
            negloglik_penalty_cell{i,j}=negloglik_penalty_new_record;
            G_record_cell{i,j}=G_record;
            U_record_cell{i,j}=U_record;
            V_record_cell{i,j}=V_record;
            W_record_cell{i,j}=W_record;
            b_record_cell{i,j}=b_record;
        end      
    end
        
    tmp=[];
    for a=1:nRank; tmp=[tmp repmat(a,1,nLambda)]; end
    indexV = [tmp' repmat(1:nRank,1,nLambda)'];
 
    %% condition for small sample size <= 200
    AIC2(AIC2<=-1000) = inf;

    [M1,I] = min(AIC2,[],1);
    Bout = BetaCell{I(rr),rr};
    negloglik_penalty_out = negloglik_penalty_cell{I(rr),rr};
    G_record_out = G_record_cell{I(rr),rr};
    U_record_out = U_record_cell{I(rr),rr};
    V_record_out = V_record_cell{I(rr),rr};
    W_record_out = W_record_cell{I(rr),rr};
    b_record_out = b_record_cell{I(rr),rr};  
    
    AICc_rslt = [AICc_rslt; M1(1,rr)];

    Idx1out=sum(Bout.U{1},2)==0;
    Idx2out=sum(Bout.U{2},2)==0;
       
    TN=sum((Idx1org==Idx1out).*Idx1org) + sum((Idx2org==Idx2out).*Idx2org); 
    TP=sum((sum(Bout.U{1},2)~=0).*(sum(beta.U{1},2)~=0)) + sum((sum(Bout.U{2},2)~=0).*(sum(beta.U{2},2)~=0)); 
    Yzero=sum(sum(beta.U{1},2)==0) + sum(sum(beta.U{2},2)==0);
    Nzero=sum(sum(beta.U{1},2)~=0) + sum(sum(beta.U{2},2)~=0);
    FN=Nzero-TP;
    FP=Yzero-TN;
        
    Acrcy=100*(TP+TN)/(Yzero+Nzero);
    
    iReport=[TP TN FN FP Acrcy];
    rslt=[rslt; iReport];

    %% Save results 
    % save(['./Simulation_result/Benchmark/Record/Result_BC1_' ImportData '_' DataName num2str(NN) '_noise' num2str(noiselevel) '_iTry' num2str(iTry) '.mat'])
end 
elapsed_time = toc;

%% Save results 
save(['./Simulation_result/Benchmark/Result_BC1_' ImportData '_' DataName num2str(NN) '_noise' num2str(noiselevel) '.mat'])

