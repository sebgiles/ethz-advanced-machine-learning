function T = imputeByColumnMedian(T)
    A = T.Variables;
    medians = nanmedian(A);
    to_replace = isnan(A).*medians;
    A(isnan(A)) = to_replace(isnan(A));
    T.Variables = A;
end
