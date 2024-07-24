function nll = bc_nll_fun_b(U,V,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xmat = X{m};
    Xi = Xmat*V';
    temp = - ( Y(m)*(b+U(:)'*Xi(:)) - log(1+exp(b+U(:)'*Xi(:))) );
    nll = nll + temp;
end

end 