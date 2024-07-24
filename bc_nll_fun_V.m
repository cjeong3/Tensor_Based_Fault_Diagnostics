function nll = bc_nll_fun_V(U,V,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xmat = X{m};
    Xi = U'*Xmat;
    temp = - ( Y(m)*(b+V(:)'*Xi(:)) - log(1+exp(b+V(:)'*Xi(:))) );
    nll = nll + temp;
end

end 