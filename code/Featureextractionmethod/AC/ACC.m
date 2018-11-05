function [Matrix]=ACC(proteinA,proteinB,OriginData );
% load data1
% load OriginData
OriginData=OriginData';
% proteinA=PositiveA{1};
% proteinB=PositiveA{2}
% OriginData = dlmread('Descriptors.csv',',');
% property.csv is the file for listing the normalized values of seven descriptors of amino acids.
AAindex = 'ACDEFGHIKLMNPQRSTVWY';
Pse=[];
proteinA= strrep(proteinA,'X','');
proteinB= strrep(proteinB,'X','');
L1=length(proteinA); 
L2=length(proteinB);
AAnum1= [];
AAnum2= [];
for i=1:L1
AAnum1 = [AAnum1,OriginData(:,findstr(AAindex,proteinA(i)))];
end
for i=1:L2
AAnum2 = [AAnum2,OriginData(:,findstr(AAindex,proteinB(i)))];
end
Matrix1=[];
Matrix2=[];
for i=1:7
    t1=zeros(1,11);
    t2=zeros(1,11);
    for j=1:11
        for k=1:(L1-j)
           J=(AAnum1(i,k)-sum(AAnum1(i,:)/L1))*(AAnum1(i,(k+j))-sum(AAnum1(i,:)/L1));
            t1(j)=t1(j)+J;
            J=[];
        end
        for k=1:(L2-j)
           J=(AAnum2(i,k)-sum(AAnum2(i,:)/L2))*(AAnum2(i,(k+j))-sum(AAnum2(i,:)/L2));
            t2(j)=t2(j)+J;
            J=[];
        end
        t1(j)=t1(j)/(L1-j);
        t2(j)=t2(j)/(L2-j);
     end
       Matrix1=[Matrix1,t1];
       Matrix2=[Matrix2,t2];
end
Matrix=[Matrix1,Matrix2];