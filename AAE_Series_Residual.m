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
settings.lrD = 0.0002;
settings.lrEn1 = 0.0002;
settings.lrDe1 = 0.0002;
settings.lrDe2 = 0.0002;
settings.lrDe3 = 0.0002;
% settings.lrEn1 = 0.0002;
% settings.lrDe1 = 0.0002;
% settings.lrDe2 = 0.0004;
% settings.lrDe3 = 0.0004;
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

%% Discriminator
paramsDis0.FCW1 = dlarray(initializeGaussian([512,settings.latent_dim],.02));
paramsDis0.FCb1 = dlarray(zeros(512,1,'single'));
paramsDis0.FCW2 = dlarray(initializeGaussian([256,512]));
paramsDis0.FCb2 = dlarray(zeros(256,1,'single'));
paramsDis0.FCW3 = dlarray(initializeGaussian([1,256]));
paramsDis0.FCb3 = dlarray(zeros(1,1,'single'));

% average Gradient and average Gradient squared holders
avgG0.Dis = [];
avgGS0.Dis = [];
avgG1.En = [];
avgGS1.En = [];
avgG1.De = [];
avgGS1.De = [];
avgG2.De = [];
avgGS2.De = [];
avgG3.De = [];
avgGS3.De = [];

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

        [GradEn1,GradDe1,GradDis0] = dlfeval(@modelGradients1,XBatch1,paramsEn1,paramsDe1,paramsDis0);

        % Update Discriminator network parameters
        [paramsDis0,avgG0.Dis,avgGS0.Dis] = adamupdate(paramsDis0,GradDis0,avgG0.Dis,avgGS0.Dis,global_iter,settings.lrD,settings.beta1,settings.beta2);

        % Update Encoder network parameters
        [paramsEn1,avgG1.En,avgGS1.En] = adamupdate(paramsEn1,GradEn1,avgG1.En,avgGS1.En,global_iter,settings.lrEn1,settings.beta1,settings.beta2);
        
        % Update Decoder network parameters
        [paramsDe1,avgG1.De,avgGS1.De] = adamupdate(paramsDe1,GradDe1,avgG1.De,avgGS1.De,global_iter,settings.lrDe1,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec1(epoch,i,XBatch1,paramsEn1,paramsDe1);
        end
      
    end
    
    %% trainX2
    % trainX1=gpdl(single(trainX1),'CB');
    y = Encoder(trainX1,paramsEn1);
    trainX2 = Decoder(y,paramsDe1);
    if epoch == settings.maxepochs-1
        trainR2 = trainX1-trainX2;
        tp = trainR2 > +1;
        tn = trainR2 < -1;
        trainR2 = trainR2 .* (1-tp) .* (1-tn) + tp - tn; 
    end
    disp("Serial1 SSIM (train)="+gatext(my_ssim(trainX2,trainX1)));

    %% testX2
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
    trainXshuffle1 = trainR2(:,shuffleid);
    trainXshuffle2 = trainX2(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch1=gpdl(single(trainXshuffle1(:,idx)),'CB');
        XBatch2=gpdl(single(trainXshuffle2(:,idx)),'CB');

        [GradDe2] = dlfeval(@modelGradients2,XBatch1,XBatch2,paramsDe2);
     
        % Update Decoder network parameters
        [paramsDe2,avgG2.De,avgGS2.De] = adamupdate(paramsDe2,GradDe2,avgG2.De,avgGS2.De,global_iter,settings.lrDe2,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec2(epoch,i,XBatch1,XBatch2,paramsDe2);
        end
      
    end

    % trainX3
    trainX3 = Decoder(trainX2,paramsDe2);
    trainX3 = trainX2+trainX3;
    tp = trainX3 > +1;
    tn = trainX3 < -1;
    trainX3 = trainX3 .* (1-tp) .* (1-tn) + tp - tn; 
    if epoch == settings.maxepochs-1
        trainR3 = trainX1-trainX3;
        tp = trainR3 > +1;
        tn = trainR3 < -1;
        trainR3 = trainR3 .* (1-tp) .* (1-tn) + tp - tn; 
    end
    disp("Serial2 SSIM (train)="+gatext(my_ssim(trainX3,trainX1)));

    % testX3
    testX3 = Decoder(testX2,paramsDe2);
    testX3 = testX2+testX3;
    tp = testX3 > +1;
    tn = testX3 < -1;
    testX3 = testX3 .* (1-tp) .* (1-tn) + tp - tn; 
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
    trainXshuffle1 = trainR3(:,shuffleid);
    trainXshuffle2 = trainX3(:,shuffleid);
    fprintf('Epoch %d\n',epoch) 
    for i=1:numIterations
        global_iter = global_iter+1;
        idx = (i-1)*settings.batch_size+1:i*settings.batch_size;
        XBatch1=gpdl(single(trainXshuffle1(:,idx)),'CB');
        XBatch2=gpdl(single(trainXshuffle2(:,idx)),'CB');

        [GradDe3] = dlfeval(@modelGradients3,XBatch1,XBatch2,paramsDe3);
     
        % Update Decoder network parameters
        [paramsDe3,avgG3.De,avgGS3.De] = adamupdate(paramsDe3,GradDe3,avgG3.De,avgGS3.De,global_iter,settings.lrDe3,settings.beta1,settings.beta2);
        
        % Compute SSIM
        if i==1 || rem(i,20)==0
            progressplot_rec3(epoch,i,XBatch1,XBatch2,paramsDe3);
        end
      
    end

    % trainX4
    trainX4 = Decoder(trainX3,paramsDe3);
    trainX4 = trainX4+trainX3;
    tp = trainX4 > +1;
    tn = trainX4 < -1;
    trainX4 = trainX4 .* (1-tp) .* (1-tn) + tp - tn; 
    disp("Serial3 SSIM (train)="+gatext(my_ssim(trainX4,trainX1)));

    % testX4
    testX4 = Decoder(testX3,paramsDe3);
    testX4 = testX4+testX3;
    tp = testX4 > +1;
    tn = testX4 < -1;
    testX4 = testX4 .* (1-tp) .* (1-tn) + tp - tn; 
    disp("Serial3 SSIM (test)="+gatext(my_ssim(testX4,testX1)));

    elapsedTime = toc;
    disp("Epoch "+epoch+". Time taken for epoch = "+elapsedTime + "s")
    epoch = epoch+1;
    if epoch == settings.maxepochs
        out = true;
    end    
end

%% Diary off
diary off;

%% Warning on
warning on;

%% Helper Functions

%% model Gradients 1
function [GradEn,GradDe,GradDis0]=modelGradients1(x,paramsEn,paramsDe,paramsDis0)
y = Encoder(x,paramsEn);
x_ = Decoder(y,paramsDe);
input_fake0 = y;
input_real0 = gpdl(randn(size(y)),'CB');
d_output_fake0 = Discriminator(input_fake0,paramsDis0);
d_output_real0 = Discriminator(input_real0,paramsDis0);
d_loss0 = -.5*mean(log(d_output_real0+eps)+log(1-d_output_fake0+eps));
g_loss = .999*mean(mean(.5*(x_-x).^2,1))+.001*mean(log(1-d_output_fake0+eps));
[GradEn,GradDe] = dlgradient(g_loss,paramsEn,paramsDe,'RetainData',true);
GradDis0 = dlgradient(d_loss0,paramsDis0);
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

%% Discriminator
function dly = Discriminator(dlx,params)
dly = fullyconnect(dlx,params.FCW1,params.FCb1);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW2,params.FCb2);
dly = leakyrelu(dly,.2);
dly = fullyconnect(dly,params.FCW3,params.FCb3);
dly = sigmoid(dly);
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

