function nll = tucker_nll_fun_b(Gmat,U,V,W,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xtenmat = tenmat(X{m},1);
    Xmat = Xtenmat.data;
    Xi = U'*Xmat*kron(W,V);
    temp = - ( Y(m)*(b+Gmat(:)'*Xi(:)) - log(1+exp(b+Gmat(:)'*Xi(:))) );
    nll = nll + temp;
end

end 