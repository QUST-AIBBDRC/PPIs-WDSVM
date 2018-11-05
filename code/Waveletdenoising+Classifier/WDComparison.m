clear all
clc
%导入数据

load('H278.mat');
ss=s(:,2:end);
C=[];
xulieshu=2867 ;
%选择小波函数和分解层数
wname='db8';
lev=6; 
for i=1:2916
x=ss(i,:);
[thr,sorh,keepapp]=ddencmp('den','wv',x);
xd3=wdencmp('gbl',x,wname,lev,thr,'h',1);
C(i,:)=xd3;
 end
% %二维降噪
[thr,sorh,keepapp] = ddencmp('den','wv',ss);
 clean = wdencmp('gbl',ss,wname,lev,thr,'h',keepapp);

 %画图 
 yiwei=C(xulieshu,:);
 erwei= clean(xulieshu,:);
 yuanshi=ss(xulieshu,:);
 
%   yiwei1=C(45,:);
%  erwei1= clean(45,:);
%  yuanshi1=ss(45,:);
 
subplot(311);plot(yuanshi,'r');title('The original signal');
% hold on
% subplot(311);plot(yuanshi1,'r');title('The original signal');

subplot(312);plot(yiwei,'r');title('1-D wavelet denoising');
% hold on
% subplot(312);plot(yiwei1,'r');title('1-D wavelet denoising');

subplot(313);plot(erwei,'r');title('2-D wavelet denoising');
% hold on
% subplot(313);plot(erwei1,'r');title('2-D wavelet denoising');

