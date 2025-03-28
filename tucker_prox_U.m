function prox = tucker_prox_U(U,lambda,gamma,tau)

prox = 0;
p1 = size(U,1); % demension-1
for p = 1:p1
    if norm(U(p,:),2) > lambda*gamma*tau
        U(p,:) = U(p,:) - lambda*gamma*tau*U(p,:)/norm(U(p,:),2);  
    else 
        U(p,:) = 0;
    end
end
prox = U;

end 