clear all
clc
load Matine_Data.mat
num1=numel(P_protein_a);
result_1=[];
result_11=[];
result_2=[];
result_22=[];
lambda=0;
 for i=1:num1
     result1=PAAC(P_protein_a{i},lambda);
     result11=PAAC(P_protein_b{i},lambda);
     result_1=[result_1;result1];
     result1=[];
      result_11=[result_11;result11];
    result11=[];
 end
  for i=1:num1
     result2=PAAC(N_protein_a{i},lambda);
     result22=PAAC(N_protein_b{i},lambda);
     result_2=[result_2;result2];
     result2=[];
     result_22=[result_22;result22];
    result22=[];
  end
 


