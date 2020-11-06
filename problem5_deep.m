%% Problem 5 : Step 1.
clear, clc

% data setting
rng('default')
x=rand(1,100).*5; xtrain=x(1:60); xtest=x(61:end);
y=@(x) 1+x+x.^2+x.^3; ytrain=y(xtrain); ytest=y(xtest);
m1=length(xtrain); m2=length(xtest);

% parameters
lambda=10; epsilon=0.001;

% feedforward network
W1=rand; W2=rand; W3=rand; b1=rand; b2=rand; b3=rand; % initial parameters
sigma=@(x) 1./(1+exp(-x));
f=@(x,W1,W2,W3,b1,b2,b3) W3*sigma(W2*sigma(W1*x+b1)+b2)+b3;

% gradient descent method for cost function
for i=1:1000
    for j =1:m1
        w1=W1-epsilon*((-2*sum(ytrain(j)-f(xtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)))*1*W2*...
            sigma(W1*sum(xtrain(j))+b1)*(1-sigma(W1*sum(xtrain(j))+b1))*1*sum(xtrain(j)) + lambda*2*W1);
        
        w2=W2-epsilon*((-2*sum(ytrain(j)-f(xtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)))*1*...
            sigma(W1*sum(xtrain(j))+b1) + lambda*2*W2);
        
        w3=W3-epsilon*((-2*sum(ytrain(j)-f(xtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2) + lambda*2*W3);
        
        B1=b1-epsilon*((-2*sum(ytrain(j)-f(xtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)))*1*W2*...
            sigma(W1*sum(xtrain(j))+b1)*(1-sigma(W1*sum(xtrain(j))+b1))*1);
        
        B2=b2-epsilon*((-2*sum(ytrain(j)-f(xtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xtrain(j))+b1)+b2)))*1);
        
        B3=b3-epsilon*((-2*sum(ytrain(j)-f(xtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1);
        
        W1=w1; W2=w2; W3=w3; b1=B1; b2=B2; b3=B3;
        
        cost1(i)=(ytrain-f(xtrain,W1,W2,W3,b1,b2,b3))*(ytrain-f(xtrain,W1,W2,W3,b1,b2,b3))'/m1 + lambda*(W1*W1'+W2*W2'+W3*W3');
        cost2(i)=(ytest-f(xtest,W1,W2,W3,b1,b2,b3))*(ytest-f(xtest,W1,W2,W3,b1,b2,b3))'/m1;
    end
end

figure(1)
plot(cost1,'LineWidth',2); grid
xlabel('Iteration'); ylabel('Cost function');
hold on
plot(cost2,'LineWidth',2)
legend('training','validation')
hold off

%% Problem 5 : Step 2.

% data setting
xsubtrain=xtrain(1:40); xsubtest=xtrain(41:end);
ysubtrain=y(xtrain(1:40)); ysubtest=y(xtrain(41:end));
m3=length(xsubtrain); m4=length(xsubtest);

rng('default')
W1=rand; W2=rand; W3=rand; b1=rand; b2=rand; b3=rand;

% gradient descent method for cost function
p=0;
stopping=min((ysubtest(1)-f(xsubtest(1),W1,W2,W3,b1,b2,b3))*(ysubtest(1)-f(xsubtest(1),W1,W2,W3,b1,b2,b3))'/m3);
for i=1:1000
    for j =1:m3
        w1=W1-epsilon*((-2*sum(ytrain(j)-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)))*1*W2*...
            sigma(W1*sum(xsubtrain(j))+b1)*(1-sigma(W1*sum(xsubtrain(j))+b1))*1*sum(xsubtrain(j)) + lambda*2*W1);
        
        w2=W2-epsilon*((-2*sum(ytrain(j)-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)))*1*...
            sigma(W1*sum(xsubtrain(j))+b1) + lambda*2*W2);
        
        w3=W3-epsilon*((-2*sum(ytrain(j)-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2) + lambda*2*W3);
        
        B1=b1-epsilon*((-2*sum(ytrain(j)-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)))*1*W2*...
            sigma(W1*sum(xsubtrain(j))+b1)*(1-sigma(W1*sum(xsubtrain(j))+b1))*1);
        
        B2=b2-epsilon*((-2*sum(ytrain(j)-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1*W3*...
            (sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)*(1-sigma(W2*sigma(W1*sum(xsubtrain(j))+b1)+b2)))*1);
        
        B3=b3-epsilon*((-2*sum(ytrain(j)-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))/m1)*1);
        
        W1=w1; W2=w2; W3=w3; b1=B1; b2=B2; b3=B3;
        
        cost1(i)=(ysubtrain-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))*(ysubtrain-f(xsubtrain(j),W1,W2,W3,b1,b2,b3))'/m3 + lambda*(W1*W1'+W2*W2'+W3*W3');
        cost2(i)=(ysubtest-f(xsubtest,W1,W2,W3,b1,b2,b3))*(ysubtest-f(xsubtest,W1,W2,W3,b1,b2,b3))'/m3;
      
    end
end
    figure(2)
    plot(cost1,'LineWidth',2); grid
    xlabel('Iteration'); ylabel('Cost function');
    hold on
    plot(cost2,'LineWidth',2)
    legend('subtraining','subvalidation')
    hold off