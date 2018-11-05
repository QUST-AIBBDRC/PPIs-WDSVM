clear all
clc
% load Matine_Data
% A1=P_protein_a;
% A2=P_protein_b;
% A3=N_protein_a;
% A4=N_protein_b;
% num_P=numel(A1);
% num_N=numel(A3);
load N_proteinA;
 load N_proteinB;
 load P_proteinA;
 load P_proteinB;
 A3=proteinA;
 A4=proteinB;
 A1=P_proteinA;
 A2=P_proteinB;
 num_P=numel(A1);
 num_N=numel(A3);
%上面是提取每条序列，下面调用选用sequence(1,:)
L=10;
for i=1:num_P
eb1_Pa(i,:)= ebgw1(A1{i},L);
eb2_Pa(i,:)= ebgw2(A1{i},L);
eb3_Pa(i,:)= ebgw3(A1{i},L);
end
for i=1:num_P
eb1_Pb(i,:)= ebgw1(A2{i},L);
eb2_Pb(i,:)= ebgw2(A2{i},L);
eb3_Pb(i,:)= ebgw3(A2{i},L);
end
for i=1:num_N
eb1_Na(i,:)= ebgw1(A3{i},L);
eb2_Na(i,:)= ebgw2(A3{i},L);
eb3_Na(i,:)= ebgw3(A3{i},L);
end
for i=1:num_N
eb1_Nb(i,:)= ebgw1(A4{i},L);
eb2_Nb(i,:)= ebgw2(A4{i},L);
eb3_Nb(i,:)= ebgw3(A4{i},L);
end
Pa=[eb1_Pa,eb2_Pa,eb3_Pa];
Pb=[eb1_Pb,eb2_Pb,eb3_Pb];
Na=[eb1_Na,eb2_Na,eb3_Na];
Nb=[eb1_Nb,eb2_Nb,eb3_Nb];

 p=[Pa,Pb];
 n=[Na,Nb];
 EBGWY10=[p;n];
save EBGW_Y_10.mat EBGWY10

