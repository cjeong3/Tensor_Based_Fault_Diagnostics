function prox = bc3_prox_B(B,lambda,gamma,tau)

prox = 0;
p2 = size(B,2); % demension-2
for p = 1:p2
    if sqrt(sum(sum(B(:,p,:).^2))) > lambda*gamma*tau
        B(:,p,:) = B(:,p,:) - lambda*gamma*tau*B(:,p,:)/sqrt(sum(sum(B(:,p,:).^2)));  
    else 
        B(:,p,:) = 0;
    end
end
prox = B;

end 