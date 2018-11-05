% clear all
% close all
%  clc  
%  
%  s1=xlsread('Y_X_NB');
%  s2=xlsread('Y_Y_NB');
% [tpr,fpr,thresholds] =roc(s2(:,1)',s1(:,1)');
% x=fpr';
% y=tpr';
% plot(x,y);
% x1=x;
% y1=y;
% xlabel('False positive rate');
% ylabel('True positive rate');
%  %save Y_NB_1 x1 y1
 
 clear all
close all
 clc  
%x3=[0 0.078 1];
%y3=[0 0.894 1];
x3=[0 0.098 1];
y3=[0 0.899 1];
  load('Y_NB_1.mat');
  load('Y_LR_2.mat');
  load('Y_Ada_4.mat');
   load('Y278_0.mat');
  subplot(121), plot(x1,y1,'-g');
   hold on
 subplot(121), plot(x2,y2,'-b');
   hold on
  subplot(121) ,plot(x3,y3,'m');
   hold on
 subplot(121) ,plot(x4,y4,'-c');
   hold on
 subplot(121),plot(x,y,'-r');
% legend('NB (AUC=0.9152)','LR (AUC=0.9275)','DT (AUC=0.9174)','Adaboost (AUC=0.9852)','SVM (AUC=0.9931)');
 legend('NB (AUC=0.8717)','LR (AUC=0.9213)','DT (AUC=0.9003)','Adaboost (AUC=0.9618)','SVM (AUC=0.9914)');
xlabel('1-Specificity');
	ylabel('Sensitivity');

 x3=[1 0.9012 0];
y3=[0.5 0.9013 1];
%x3=[0 0.098 1];
%y3=[0 0.899 1];
  NB=xlsread('YNB.xlsx');
  x1=NB(:,1);
   y1=NB(:,2);
  LR=xlsread('YLR.xlsx');
   x2=LR(:,1);
   y2=LR(:,2); 
 ADA=xlsread('YTF_ADA.xlsx');
   x4=ADA(:,1);
   y4=ADA(:,2); 
  SVM=xlsread('YSVM.xlsx');
   x=SVM(:,1);
   y=SVM(:,2); 
 subplot(122), plot(x1,y1,'-g');
   hold on
 subplot(122), plot(x2,y2,'-b');
   hold on
   subplot(122),plot(x3,y3,'m');
   hold on
   subplot(122),plot(x4,y4,'-c');
   hold on
subplot(122), plot(x,y,'-r');
 %legend('NB (AUPR=0.9100)','LR (AUPR=0.9357)','DT (AUPR=0.8887)','Adaboost (AUPR=0.9859)','SVM (AUPR=0.9920)');
 legend('NB (AUPR=0.8390)','LR (AUPR=0.9183)','DT (AUPR=0.8601)','Adaboost (AUPR=0.9638)','SVM (AUPR=0.9916)');
  xlabel('Sensitivity');
	ylabel('Precision');


