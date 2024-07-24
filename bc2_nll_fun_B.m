function nll = bc2_nll_fun_B(B,b,X,Y,NN)

nll = 0;
for m = 1:NN
    Xmat = double(X{m});
    temp = - ( Y(m)*(b+B(:)'*Xmat(:)) - log(1+exp(b+B(:)'*Xmat(:))) );
    nll = nll + temp;
end

end 
