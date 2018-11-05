 clear all
 clc
% load PAACY9.mat;
% load AC_Y_11.mat;
% load EBGW_Y_11.mat;
% s1=[PAACY9,ACCY154];
% s2=[s1,EBGWY11];
% s3=xlsread('11188');
% s=[s3,s2];
% save Y278.mat s
load Y278.mat;
 yuanshu=s(:,2:end);
%%%%%%%%%%%%%%%%%
%二维小波降噪ddencmp默认阈值
[thr,sorh,keepapp] = ddencmp('den','wv',yuanshu);
clean = wdencmp('gbl',yuanshu,'db8',6,thr,'h',keepapp);
yuanshu=clean;
%%%%%%%%%%%%%%%%%
% %使用Birge-Massart策略降噪
% wname='db3';
% lev=5; 
% [c,l]=wavedec2(yuanshu,lev,wname); 
% sigma=wnoisest(c,l,1); 
% alpha=5;
% [thr,nkeep]=wdcbm2(c,l,alpha);
% [clean,cxd,lxd,perf0,perfl2]=wdencmp('lvd',yuanshu,wname,lev,thr,'h'); 
% db=clean;
% yuanshu=db;  
% % %%%%%%%%%%%%%%%%%%%%%
% load H.mat;
% s=H;
% yuanshu=s(:,2:end);

shu=zscore(yuanshu);
label=s(:,1);
num=5; %五折交叉验证
data_1=[shu,label];
% data=suiji1(dataa);
[M,N]=size(data_1);
indices=crossvalind('Kfold',M,num);
testlabel=[];yucelabel=[];
 for k=1:num  %交叉验证
        indice1 = (indices == k);%获得test集元素在数据集中对应的单元编号
        indice2= ~indice1;%train集元素的编号为非test元素的编号
        train_shu=data_1(indice2,1:(N-1));
        train_shu=zscore(train_shu);
        test_shu=data_1(indice1,1:(N-1));
        test_shu=zscore(test_shu);
        train_label=data_1(indice2,N);
        test_label=data_1(indice1,N);
        model=svmtrain(train_label,train_shu,'-t 2');
    [predict_label,accuracy]=svmpredict(test_label,test_shu,model);
  yucelabel=[yucelabel;predict_label];
  testlabel=[testlabel;test_label];
  train_shu=[];test_shu=[];train_label=[];test_label=[];predict_label=[];
 end
 
 [SE,Pre,ACC,MCC,tp,tn]=VF(testlabel,yucelabel);

 


 