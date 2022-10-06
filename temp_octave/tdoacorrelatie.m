
close;
clear;
pkg load signal;
[audio , freq]=audioread("av.flac",'native');

 % 10 x hogere frqeuntie dan freq
resam=20;

shift=155;
aantalsamples=50000;
audio=audio(100000:100000+aantalsamples,1);
audio=resample(audio,resam , 1);
a= audio(5000:5000+aantalsamples);
b= audio(5000+shift:5000+aantalsamples+shift);
%b=b/10+100*randn(size(b));


figure(1);
plot(a);
waitforbuttonpress ()

figure(2);
plot(b);
waitforbuttonpress ()

%figure(2);
%plot(f);
%hold on
lag=aantalsamples/5;
[rxy,lags]=xcorr(a,b,lag );

%hold on;
figure(3);
plot(lags,rxy,'g');
waitforbuttonpress ()

[x,aa] = max(rxy);
figure(4);
plot(rxy(lag:lag+10000));
waitforbuttonpress ()