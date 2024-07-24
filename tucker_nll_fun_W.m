function nll = tucker_nll_fun_W(Gmat,U,V,W,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xtenmat = tenmat(X{m},3);
    Xmat = Xtenmat.data;
    Xi = Xmat*kron(V,U)*Gmat';
    temp = - ( Y(m)*(b+W(:)'*Xi(:)) - log(1+exp(b+W(:)'*Xi(:))) );
    nll = nll + temp;
end

end 