clear all
clc
load Matine_Data.mat
load OriginData
PositiveA=P_protein_a;
PositiveB=P_protein_b;
NegativeA=N_protein_a;
NegativeB=N_protein_b;
% function [result]=AAC_file(data)
num1=numel(PositiveA);
result_1=[];
result_2=[];
% num1=3;
 for i=1:num1
     result1=ACC(PositiveA{i},PositiveB{i},OriginData);
     result_1=[result_1;result1];
     result1=[];
 end
  for i=1:num1
     result2=ACC(NegativeA{i},NegativeB{i},OriginData);
     result_2=[result_2;result2];
     result2=[];
  end
%  
%   ACCY14=[result_2;result_1];
%   nu=xlsread('1122');
%   ACCY14=[nu,ACCY14];
%   xlswrite('C:\Users\appe\Desktop\ACCY56.xls',ACCY14);
  
  