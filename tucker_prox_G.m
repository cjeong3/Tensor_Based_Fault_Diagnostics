function prox = tucker_prox_G(G,lambda,tau)

prox = 0;
p1 = size(G,1); % demension-1
p2 = size(G,2); % demension-2
for p = 1:p1
    for q = 1:p2
        if G(p,q) > lambda*tau
            G(p,q) = G(p,q) - lambda*tau;
        elseif G(p,q) < -lambda*tau 
            G(p,q) = G(p,q) + lambda*tau;
        else
            G(p,q) = 0;
        end
    end     
end
prox = G;

end 

