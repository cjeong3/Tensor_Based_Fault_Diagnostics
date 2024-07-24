function nll = cp_nll_fun_U(U,V,W,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xtenmat = tenmat(X{m},1);
    Xmat = Xtenmat.data;
    Xi = Xmat*khatrirao(W,V);
    temp = - ( Y(m)*(b+U(:)'*Xi(:)) - log(1+exp(b+U(:)'*Xi(:))) );
    nll = nll + temp;
end

end 