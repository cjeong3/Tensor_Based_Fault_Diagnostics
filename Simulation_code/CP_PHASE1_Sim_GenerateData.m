%% SIMULATION STUDY

% #########################################################################
% Title: (PHASE1) Generate Data
% Output Path: /Simulation_data
% #########################################################################

clc; clear;
addpath('./tensor_toolbox-v3.1')

%% Generate X_i
rng('default') 

% Data1
n=300; p1=7; p2=8; p3=5; r=3; D=3; nTry=30; 
Index1=[1 3 5 7]; Index2=[2 4 6 8];

% Data2
% n=300; p1=10; p2=10; p3=20; r=3; D=3; nTry=30; 
% Index1=[1 3 5 7 9]; Index2=[2 4 6 8 10];

pvec=[p1 p2 p3];

% 1. IID 
rng('default') 

for i=1:nTry
    XXXX=cell(n,1);
    for j=1:n
        for k=1:p1
            XX=randn(p2,p3);
            XXX(k,:,:)=XX;
        end
        XXXX{j} = tensor(XXX);
    end
    XA{i}=XXXX; 
end
clear XX XXX XXXX;

% 2. Row Correlation: Process Variable Correlation
rng('default') 

mu=zeros(1,p2);
sigma=zeros(p2,p2);

for i=1:p2
    for j=1:p2
        sigma(i,j)=0.5^abs(i-j); % entry of coveriance matrix sigma 
    end
end
for i=1:nTry
    XXXX=cell(n,1);
    for j=1:n
        for k=1:p1
            XX=mvnrnd(mu,sigma,p3)';
            XXX(k,:,:)=XX;
        end
        XXXX{j} = tensor(XXX);
    end
    XB{i}=XXXX; 
end
clear XX XXX XXXX sigma;

%% Simulate coefficient tensor
% 1. Simulate coefficient tensor B with rank 3
rng('default') 

U=2*rand(p1,r)-1;
V=2*rand(p2,r)-1;
W=2*rand(p3,r)-1;
U(Index1,:)=zeros(length(Index1),r);
V(Index2,:)=zeros(length(Index2),r);

beta=ktensor(arrayfun(@(j) 1-2*rand(pvec(j),r), 1:3,'UniformOutput',false));
beta.U{1}=U;
beta.U{2}=V;
beta.U{3}=W;
vecBeta=reshape(double(tensor(beta)),1,p1*p2*p3);
% The above code can be replaced by % B = khatrirao(B3,B2,B1); vecB=B*ones(r1,1);

% 2. Noise
for i=1:nTry
    XXXX=cell(n,1);
    for j=1:n
        for k=1:p1
            XX=randn(p2,p3);
            XXX(k,:,:)=XX;
        end
        XXXX{j} = tensor(XXX);
    end
    noisetensor{i}=XXXX; 
end
clear XX XXX XXXX;

%% Save data
save(['./Simulation_data/DataCP1_' num2str(n) '.mat'])
% save(['./Simulation_data/DataCP2_' num2str(n) '.mat'])
