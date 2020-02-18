%% load data %%
clear all;
close all;

%Load Fisher vectors 
% X = load('simulation_1_fixed.mp4_BRISK.csv_st6_FVs.csv');
% save('simulation_1_fixed.mp4_BRISK.csv_st6_FVs.mat','X','-v7.3');


%% simmulation_1, BRISK, FV encoding 20 clusters
%Load annotations 
filename = 'simulation_1_fixed.mp4_BRISK.csv_st6_ANN.csv';
delimiter = ',';
formatSpec = '%C%C%C%C%C%C%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);
fclose(fileID);
Y = table(dataArray{1:end-1}, 'VariableNames', {'attr_a','attr_b','attr_c','attr_d','attr_e','attr_f'});
clearvars filename delimiter formatSpec fileID dataArray ans;

save('simulation_1_fixed.mp4_BRISK.csv_st6_ANN.mat','Y','-v7.3');

%% frames
frames = load('simulation_1_fixed.mp4_BRISK.csv_st6_FRAMES.csv');
save('simulation_1_fixed.mp4_BRISK.csv_st6_FRAMES.mat','frames','-v7.3');