function prox = tucker_prox_W(W,lambda,tau)

prox = 0;
p1 = size(W,1); % demension-1
p2 = size(W,2); % demension-2
for p = 1:p1
    for q = 1:p2
        if W(p,q) > lambda*tau
            W(p,q) = W(p,q) - lambda*tau;
        elseif W(p,q) < -lambda*tau 
            W(p,q) = W(p,q) + lambda*tau;
        else
            W(p,q) = 0;
        end
    end     
end
prox = W;

end 
