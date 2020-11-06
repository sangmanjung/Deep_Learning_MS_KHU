clear all, clc

% parameters
x=[0 0]'; % Initial guess
A=[1 0; 0 0.1^15]; b=[1 1]';
epsilon=0.1; % Learning rate
max_iter=100;
precision=0.001;
% function phi
phi=@(x) 1/2*x'*A*x-x'*b;
grad_phi=@(x) A*x-b;
fvals=[];
% Gradient descent method
x_new=x;
x_old=x_new;
for i=1:max_iter
    x_new=x_old-epsilon.*grad_phi(x_old);
    iter=i;
    if (norm(grad_phi(x_new))<=precision)
        break
    end
    x_old=x_new;
    xvals(iter,:)=x_new;
    fvals(iter)=phi(x_new);
end
xvals
fvals

% Graph
s=-10:10; t=0:10;
[X,Y]=meshgrid(s,t);
Z=X.^2/2+0.1^15*Y.^2/2-X-Y;
figure(1)
contour(X,Y,Z,'ShowText','on')
hold on
plot3(xvals(:,1),xvals(:,2),fvals,'r-o','LineWidth',2)

figure(2)
plot3(xvals(:,1),xvals(:,2),fvals,'r-o','LineWidth',2)
grid on
hold on
surfc(X,Y,Z)