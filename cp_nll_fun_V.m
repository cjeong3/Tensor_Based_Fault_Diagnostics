function nll = cp_nll_fun_V(U,V,W,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xtenmat = tenmat(X{m},2);
    Xmat = Xtenmat.data;
    Xi = Xmat*khatrirao(W,U);
    temp = - ( Y(m)*(b+V(:)'*Xi(:)) - log(1+exp(b+V(:)'*Xi(:))) );
    nll = nll + temp;
end

end 