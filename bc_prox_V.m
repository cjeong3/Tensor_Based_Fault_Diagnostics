function prox = bc_prox_V(V,lambda,gamma,tau)

prox = 0;
p2 = size(V,2); % demension-1
for p = 1:p2
    if norm(V(:,p),1) > lambda*gamma*tau
        V(:,p) = V(:,p) - lambda*gamma*tau*V(:,p)/norm(V(:,p),1);  
    else 
        V(:,p) = 0;
    end
end
prox = V;

end 