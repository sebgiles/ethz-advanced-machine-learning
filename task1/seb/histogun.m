function histogun(X1,X2,X3)
names = X1.Properties.VariableNames;
for i=2:length(names)
    x=names{i};
    clf
    histogram(X1(:,x).Variables,100)
    hold on

    histogram(X2(:,x).Variables,100)
    if nargin==3
        histogram(X3(:,x).Variables,100)
     	legend('1','2','3')
    else
        legend('1','2')
    end
    disp(x)
    pause()
end