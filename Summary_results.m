
% #########################################################################
% Data: Summary
% #########################################################################

clear all; close all; clc; 
addpath('./tensor_toolbox-v3.1')

% #########################################################################
% 1. NUMERICAL STUDY
% #########################################################################

ImportData_new = 'DataCP1'; % DataCP1 DataCP2 DataTucker1 DataTucker2

NN_set = [200 300 500];
noiselevel_set = [1 2 3];
DataName_set = {'XA' 'XB'};

result = [];

for DataName_idx = 1:2

    result_temp =[];
    
    for NN_new = NN_set
        for noiselevel_new = noiselevel_set       

            % load(['./Simulation_result/CP/Result_' ImportData_new '_' DataName_set{DataName_idx} num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            % % load(['./Simulation_result/CP/Result_' ImportData_new '_' DataName_new num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            % tempA = [100*rslt(:,1)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,2)./(rslt(:,4)+rslt(:,2)) 100*rslt(:,3)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,4)./(rslt(:,4)+rslt(:,2)) 100*(rslt(:,1)+rslt(:,2))./(rslt(:,1)+rslt(:,2)+rslt(:,3)+rslt(:,4))];
            % tempB = [mean(tempA,1) std(tempA,0,1)];                    
            % result_temp = [result_temp; tempB];    

            load(['./Simulation_result/Tucker/Result_' ImportData_new '_' DataName_set{DataName_idx} num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            % load(['./Simulation_result/Tucker/Result_' ImportData_new '_' DataName_new num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            tempA = [100*rslt(:,1)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,2)./(rslt(:,4)+rslt(:,2)) 100*rslt(:,3)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,4)./(rslt(:,4)+rslt(:,2)) 100*(rslt(:,1)+rslt(:,2))./(rslt(:,1)+rslt(:,2)+rslt(:,3)+rslt(:,4))];
            tempB = [mean(tempA,1) std(tempA,0,1)];                    
            result_temp = [result_temp; tempB];                 

            load(['./Simulation_result/Benchmark/Result_BC1_' ImportData_new '_' DataName_set{DataName_idx} num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            tempA = [100*rslt(:,1)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,2)./(rslt(:,4)+rslt(:,2)) 100*rslt(:,3)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,4)./(rslt(:,4)+rslt(:,2)) 100*(rslt(:,1)+rslt(:,2))./(rslt(:,1)+rslt(:,2)+rslt(:,3)+rslt(:,4))];
            tempB = [mean(tempA,1) std(tempA,0,1)];                    
            result_temp = [result_temp; tempB];                   
            
            load(['./Simulation_result/Benchmark/Result_BC2_' ImportData_new '_' DataName_set{DataName_idx} num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            tempA = [100*rslt(:,1)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,2)./(rslt(:,4)+rslt(:,2)) 100*rslt(:,3)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,4)./(rslt(:,4)+rslt(:,2)) 100*(rslt(:,1)+rslt(:,2))./(rslt(:,1)+rslt(:,2)+rslt(:,3)+rslt(:,4))];
            tempB = [mean(tempA,1) std(tempA,0,1)];                    
            result_temp = [result_temp; tempB];              

            load(['./Simulation_result/Benchmark/Result_BC3_' ImportData_new '_' DataName_set{DataName_idx} num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            tempA = [100*rslt(:,1)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,2)./(rslt(:,4)+rslt(:,2)) 100*rslt(:,3)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,4)./(rslt(:,4)+rslt(:,2)) 100*(rslt(:,1)+rslt(:,2))./(rslt(:,1)+rslt(:,2)+rslt(:,3)+rslt(:,4))];
            tempB = [mean(tempA,1) std(tempA,0,1)];                    
            result_temp = [result_temp; tempB];              

            load(['./Simulation_result/Benchmark/Result_BC4_' ImportData_new '_' DataName_set{DataName_idx} num2str(NN_new) '_noise' num2str(noiselevel_new) '.mat'])
            tempA = [100*rslt(:,1)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,2)./(rslt(:,4)+rslt(:,2)) 100*rslt(:,3)./(rslt(:,1)+rslt(:,3)) 100*rslt(:,4)./(rslt(:,4)+rslt(:,2)) 100*(rslt(:,1)+rslt(:,2))./(rslt(:,1)+rslt(:,2)+rslt(:,3)+rslt(:,4))];
            tempB = [mean(tempA,1) std(tempA,0,1)];                    
            result_temp = [result_temp; tempB];    

        end
    end
    result = [result result_temp]; 

end

result 
