%% Clear and Close
clear all;
close all;
clc;

%% Warning off
warning off;

%% Diary on
% datafilename = 'data\mnist\mnistAll.mat';
% datafilename = 'data\emnist\emnistdigitsAll.mat';
% datafilename = 'data\emnist\emnistlettersAll.mat';
% datafilename = 'data\emnist\emnistbalancedAll.mat';
% datafilename = 'data\emnist\emnistbymergeAll.mat';
% datafilename = 'data\emnist\emnistbyclassAll.mat';
% datafilename = 'data\fashionmnist\fashionmnistAll.mat';
% datafilename = 'data\medmnist\breastmnistAll.mat';
% datafilename = 'data\medmnist\chestmnistAll.mat';
% datafilename = 'data\medmnist\dermamnistAll.mat';
% datafilename = 'data\medmnist\octmnistAll.mat';
datafilename = 'data\medmnist\organmnist_axialAll.mat';
% datafilename = 'data\medmnist\organmnist_coronalAll.mat';
% datafilename = 'data\medmnist\organmnist_sagittalAll.mat';
% datafilename = 'data\medmnist\pathmnistAll.mat';
% datafilename = 'data\medmnist\pneumoniamnistAll.mat';
% datafilename = 'data\medmnist\retinamnistAll.mat';
n1 = 1;
n2 = 1;
while datafilename(n1) ~= '\'
    n1 = n1+1;
end
n1 = n1+1;
while datafilename(n1) ~= '\'
    n1 = n1+1;
end
n1 = n1+1;
while datafilename(n1) ~= '.'
    datafilename2(n2)=datafilename(n1);
    n1 = n1+1;
    n2 = n2+1;
end
diaryfilename = ['result\' mfilename '_' datafilename2 '_' datestr(now,30) '.txt'];
diary(diaryfilename);

%% Adversarial AutoEncoder

%% Load Data
load(datafilename);
trainX1 = preprocess(mnist.train_images);
trainY1 = mnist.train_labels;
testX1 = preprocess(mnist.test_images);
testY1 = mnist.test_labels;

%% Prepare Data
trainX1=gpdl(single(trainX1),'CB');
testX1=gpdl(single(testX1),'CB');

%% Settings
% settings.latent_dim = 10;
settings.latent_dim = 30;
% settings.batch_size = 32;
settings.batch_size = 100;
settings.image_size = [28,28,1]; 
settings.lrG = 0.0002;
settings.beta1 = 0.5;
settings.beta2 = 0.999;
% settings.maxepochs = 5;
% settings.maxepochs = 50;
settings.maxepochs = 100;
% settings.maxepochs = 1000;

%% Initialization

%% Encoder
paramsEn1.FCW1 = dlarray(initializeGaussian([512,prod(settings.image_size)],.02));
paramsEn1.FCb1 = dlarray(zeros(512,1,'single'));
paramsEn1.FCW2 = dlarray(initializeGaussian([512,512]));
paramsEn1.FCb2 = dlarray(zeros(512,1,'single'));
paramsEn1.FCW3 = dlarray(initializeGaussian([settings.latent_dim,512]));
paramsEn1.FCb3 = dlarray(zeros(settings.latent_dim,1,'single'));

%% Decoder
paramsDe1.FCW1 = dlarray(initializeGaussian([512,settings.latent_dim],.02));
paramsDe1.FCb1 = dlarray(zeros(512,1,'single'));
paramsDe1.FCW2 = dlarray(initializeGaussian([512,512]));
paramsDe1.FCb2 = dlarray(zeros(512,1,'single'));
paramsDe1.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),512]));
paramsDe1.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));
% paramsDe2.FCW1 = dlarray(initializeGaussian([512,settings.latent_dim],.02));
% paramsDe2.FCb1 = dlarray(zeros(512,1,'single'));
% paramsDe2.FCW2 = dlarray(initializeGaussian([512,512]));
% paramsDe2.FCb2 = dlarray(zeros(512,1,'single'));
% paramsDe2.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),512]));
% paramsDe2.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));
paramsDe2.FCW1 = dlarray(initializeGaussian([2048,prod(settings.image_size)],.02));
paramsDe2.FCb1 = dlarray(zeros(2048,1,'single'));
paramsDe2.FCW2 = dlarray(initializeGaussian([2048,2048]));
paramsDe2.FCb2 = dlarray(zeros(2048,1,'single'));
paramsDe2.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),2048]));
paramsDe2.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));
paramsDe3.FCW1 = dlarray(initializeGaussian([2048,prod(settings.image_size)],.02));
paramsDe3.FCb1 = dlarray(zeros(2048,1,'single'));
paramsDe3.FCW2 = dlarray(initializeGaussian([2048,2048]));
paramsDe3.FCb2 = dlarray(zeros(2048,1,'single'));
paramsDe3.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),2048]));
paramsDe3.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));
paramsDe4.FCW1 = dlarray(initializeGaussian([2048,prod(settings.image_size)],.02));
paramsDe4.FCb1 = dlarray(zeros(2048,1,'single'));
paramsDe4.FCW2 = dlarray(initializeGaussian([2048,2048]));
paramsDe4.FCb2 = dlarray(zeros(2048,1,'single'));
paramsDe4.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),2048]));
paramsDe4.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));
paramsDe5.FCW1 = dlarray(initializeGaussian([2048,prod(settings.image_size)],.02));
paramsDe5.FCb1 = dlarray(zeros(2048,1,'single'));
paramsDe5.FCW2 = dlarray(initializeGaussian([2048,2048]));
paramsDe5.FCb2 = dlarray(zeros(2048,1,'single'));
paramsDe5.FCW3 = dlarray(initializeGaussian([prod(settings.image_size),2048]));
paramsDe5.FCb3 = dlarray(zeros(prod(settings.image_size),1,'single'));

% average Gradient and average Gradient squared holders
avgG1.En = [];
avgGS1.En = [];
avgG1.De = [];
avgGS1.De = [];
avgG2.De = [];
avgGS2.De = [];
avgG3.De = [];
avgGS3.De = [];
avgG4.De = [];
avgGS4.De = [];
avgG5.De = [];
avgGS5.De = [];

%% Train 1
numIterations = floor(size(trainX1,2)/settings.batch_size);
out = false;
epoch = 0;
global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX1,2));
    trainXshuffle1 = trainX1(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch1=gpdl(single(trainXshuffle1(:,idx)),'CB');

        [GradEn1,GradDe1] = dlfeval(@modelGradients1,XBatch1,paramsEn1,paramsDe1);

        % Update Encoder network parameters
        [paramsEn1,avgG1.En,avgGS1.En] = adamupdate(paramsEn1,GradEn1,avgG1.En,avgGS1.En,global_iter,settings.lrG,settings.beta1,settings.beta2);
        
        % Update Decoder network parameters
        [paramsDe1,avgG1.De,avgGS1.De] = adamupdate(paramsDe1,GradDe1,avgG1.De,avgGS1.De,global_iter,settings.lrG,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec1(epoch,i,XBatch1,paramsEn1,paramsDe1);
        end
      
    end
    
    % trainX2
    % trainX1=gpdl(single(trainX1),'CB');
    y = Encoder(trainX1,paramsEn1);
    trainX2 = Decoder(y,paramsDe1);
    disp("Serial1 SSIM (train)="+gatext(my_ssim(trainX2,trainX1)));

    % testX2
    % testX1=gpdl(single(testX1),'CB');
    y = Encoder(testX1,paramsEn1);
    testX2 = Decoder(y,paramsDe1);
    disp("Serial1 SSIM (test)="+gatext(my_ssim(testX2,testX1)));

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

%% Train 2
numIterations = floor(size(trainX2,2)/settings.batch_size);
out = false;
epoch = 0;
global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX2,2));
    trainXshuffle1 = trainX1(:,shuffleid);
    trainXshuffle2 = trainX2(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch1=gpdl(single(trainXshuffle1(:,idx)),'CB');
        XBatch2=gpdl(single(trainXshuffle2(:,idx)),'CB');

        [GradDe2] = dlfeval(@modelGradients2,XBatch1,XBatch2,paramsDe2);
     
        % Update Decoder network parameters
        [paramsDe2,avgG2.De,avgGS2.De] = adamupdate(paramsDe2,GradDe2,avgG2.De,avgGS2.De,global_iter,settings.lrG,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec2(epoch,i,XBatch1,XBatch2,paramsDe2);
        end
      
    end

    % trainX3
    trainX3 = Decoder(trainX2,paramsDe2);
    disp("Serial2 SSIM (train)="+gatext(my_ssim(trainX3,trainX1)));

    % testX3
    testX3 = Decoder(testX2,paramsDe2);
    disp("Serial2 SSIM (test)="+gatext(my_ssim(testX3,testX1)));

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

%% Parameters3
paramsDe3 = paramsDe2;
avgG3.De = avgG2.De;
avgGS3.De = avgGS2.De;

%% Train 3
numIterations = floor(size(trainX3,2)/settings.batch_size);
out = false;
epoch = 0;
global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX3,2));
    trainXshuffle1 = trainX1(:,shuffleid);
    trainXshuffle2 = trainX3(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch1=gpdl(single(trainXshuffle1(:,idx)),'CB');
        XBatch2=gpdl(single(trainXshuffle2(:,idx)),'CB');

        [GradDe3] = dlfeval(@modelGradients3,XBatch1,XBatch2,paramsDe3);
     
        % Update Decoder network parameters
        [paramsDe3,avgG3.De,avgGS3.De] = adamupdate(paramsDe3,GradDe3,avgG3.De,avgGS3.De,global_iter,settings.lrG,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec3(epoch,i,XBatch1,XBatch2,paramsDe3);
        end
      
    end
    
    % trainX4
    trainX4 = Decoder(trainX3,paramsDe3);
    disp("Serial3 SSIM (train)="+gatext(my_ssim(trainX4,trainX1)));
    
    % testX4
    testX4 = Decoder(testX3,paramsDe3);
    disp("Serial3 SSIM (test)="+gatext(my_ssim(testX4,testX1)));

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

%{

%% Parameters4
paramsDe4 = paramsDe3;
avgG4.De = avgG3.De;
avgGS4.De = avgGS3.De;

%% Train 4
numIterations = floor(size(trainX4,2)/settings.batch_size);
out = false;
epoch = 0;
global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX4,2));
    trainXshuffle1 = trainX1(:,shuffleid);
    trainXshuffle2 = trainX4(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch1=gpdl(single(trainXshuffle1(:,idx)),'CB');
        XBatch2=gpdl(single(trainXshuffle2(:,idx)),'CB');

        [GradDe4] = dlfeval(@modelGradients4,XBatch1,XBatch2,paramsDe4);
     
        % Update Decoder network parameters
        [paramsDe4,avgG4.De,avgGS4.De] = adamupdate(paramsDe4,GradDe4,avgG4.De,avgGS4.De,global_iter,settings.lrG,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec4(epoch,i,XBatch1,XBatch2,paramsDe4);
        end
      
    end
    
    % trainX5
    trainX5 = Decoder(trainX4,paramsDe4);
    disp("Serial4 SSIM (train)="+gatext(my_ssim(trainX5,trainX1)));
    
    % testX5
    testX5 = Decoder(testX4,paramsDe4);
    disp("Serial4 SSIM (test)="+gatext(my_ssim(testX5,testX1)));

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

%% Parameters5
paramsDe5 = paramsDe4;
avgG5.De = avgG4.De;
avgGS5.De = avgGS4.De;

%% Train 5
numIterations = floor(size(trainX5,2)/settings.batch_size);
out = false;
epoch = 0;
global_iter = 0;
while ~out
    tic; 
    shuffleid = randperm(size(trainX5,2));
    trainXshuffle1 = trainX1(:,shuffleid);
    trainXshuffle2 = trainX5(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch1=gpdl(single(trainXshuffle1(:,idx)),'CB');
        XBatch2=gpdl(single(trainXshuffle2(:,idx)),'CB');

        [GradDe5] = dlfeval(@modelGradients5,XBatch1,XBatch2,paramsDe5);
     
        % Update Decoder network parameters
        [paramsDe5,avgG5.De,avgGS5.De] = adamupdate(paramsDe5,GradDe5,avgG5.De,avgGS5.De,global_iter,settings.lrG,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec5(epoch,i,XBatch1,XBatch2,paramsDe5);
        end
      
    end
    
    % trainX6
    trainX6 = Decoder(trainX5,paramsDe5);
    disp("Serial5 SSIM (train)="+gatext(my_ssim(trainX6,trainX1)));
    
    % testX6
    testX6 = Decoder(testX5,paramsDe5);
    disp("Serial5 SSIM (test)="+gatext(my_ssim(testX6,testX1)));

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

%}

%% Diary off
diary off;

%% Warning on
warning on;

%% Helper Functions

%% model Gradients 1
function [GradEn,GradDe]=modelGradients1(x,paramsEn,paramsDe)
y = Encoder(x,paramsEn);
x_ = Decoder(y,paramsDe);
g_loss = mean(mean(.5*(x_-x).^2,1));
[GradEn,GradDe] = dlgradient(g_loss,paramsEn,paramsDe,'RetainData',true);
end

%% model Gradients 2
function [GradDe]=modelGradients2(xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
g_loss = mean(mean(.5*(x_-xr).^2,1));
[GradDe] = dlgradient(g_loss,paramsDe,'RetainData',true);
end

%% model Gradients 3
function [GradDe]=modelGradients3(xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
g_loss = mean(mean(.5*(x_-xr).^2,1));
[GradDe] = dlgradient(g_loss,paramsDe,'RetainData',true);
end

%% model Gradients 4
function [GradDe]=modelGradients4(xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
g_loss = mean(mean(.5*(x_-xr).^2,1));
[GradDe] = dlgradient(g_loss,paramsDe,'RetainData',true);
end

%% model Gradients 5
function [GradDe]=modelGradients5(xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
g_loss = mean(mean(.5*(x_-xr).^2,1));
[GradDe] = dlgradient(g_loss,paramsDe,'RetainData',true);
end

%% preprocess
function x = preprocess(x)
x = double(x)/255;
x = (x-.5)/.5;
x = reshape(x,28*28,[]);
end

%% extract data
function x = gatext(x)
x = gather(extractdata(x));
end

%% gpu dl array wrapper
function dlx = gpdl(x,labels)
dlx = dlarray(x,labels);
end

%% Weight initialization
function parameter = initializeGaussian(parameterSize,sigma)
if nargin < 2
    sigma = 0.05;
end
parameter = randn(parameterSize, 'single') .* sigma;
end

%% Encoder
function dly = Encoder(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,.2);
end

%% Decoder
function dly = Decoder(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = leakyrelu(dly,.2);
dly = tanh(dly);
end

%% progressplot_rec1
function progressplot_rec1(Epoch,Iter,x,paramsEn,paramsDe)
y = Encoder(x,paramsEn);
x_ = Decoder(y,paramsDe);
disp("Epoch="+Epoch+",Iter="+Iter+",SSIM="+gatext(my_ssim(x_,x)));
end

%% progressplot_rec2
function progressplot_rec2(Epoch,Iter,xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
disp("Epoch="+Epoch+",Iter="+Iter+",SSIM="+gatext(my_ssim(x_,xr)));
end

%% progressplot_rec3
function progressplot_rec3(Epoch,Iter,xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
disp("Epoch="+Epoch+",Iter="+Iter+",SSIM="+gatext(my_ssim(x_,xr)));
end

%% progressplot_rec4
function progressplot_rec4(Epoch,Iter,xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
disp("Epoch="+Epoch+",Iter="+Iter+",SSIM="+gatext(my_ssim(x_,xr)));
end

%% progressplot_rec5
function progressplot_rec5(Epoch,Iter,xr,x,paramsDe)
x_ = Decoder(x,paramsDe);
disp("Epoch="+Epoch+",Iter="+Iter+",SSIM="+gatext(my_ssim(x_,xr)));
end

%% my_ssim
function [re] = my_ssim(X,Y)
    X=double(X);
    Y=double(Y);

    ux=mean(mean(X));
    uy=mean(mean(Y));

    sigma2x=mean(mean((X-ux).^2));
    sigma2y=mean(mean((Y-uy).^2));   
    sigmaxy=mean(mean((X-ux).*(Y-uy)));

    k1=0.01;
    k2=0.03;
    %L=255;
    L=2;
    %L=1;
    c1=(k1*L)^2;
    c2=(k2*L)^2;
    c3=c2/2;

    l=(2*ux.*uy+c1)/(ux.*ux+uy.*uy+c1);
    c=(2*sqrt(sigma2x).*sqrt(sigma2y)+c2)/(sigma2x+sigma2y+c2);
    s=(sigmaxy+c3)/(sqrt(sigma2x).*sqrt(sigma2y)+c3);

    re=l.*c.*s;
end

