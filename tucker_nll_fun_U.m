function nll = tucker_nll_fun_U(Gmat,U,V,W,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xtenmat = tenmat(X{m},1);
    Xmat = Xtenmat.data;
    Xi = Xmat*kron(W,V)*Gmat';
    temp = - ( Y(m)*(b+U(:)'*Xi(:)) - log(1+exp(b+U(:)'*Xi(:))) );
    nll = nll + temp;
end

end 