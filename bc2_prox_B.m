function prox = bc2_prox_B(B,lambda,gamma,tau)

prox = 0;
p1 = size(B,1); % demension-1
for p = 1:p1
    if sqrt(sum(sum(B(p,:,:).^2))) > lambda*gamma*tau
        B(p,:,:) = B(p,:,:) - lambda*gamma*tau*B(p,:,:)/sqrt(sum(sum(B(p,:,:).^2)));  
    else 
        B(p,:,:) = 0;
    end
end
prox = B;

end 