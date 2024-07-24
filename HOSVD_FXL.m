%%This code is for HOSVD
function [Tucker,U,NPC]=HOSVD_FXL(T,FVED)
Tucker=T;
for k=1:length(T.size)
    C=tenmat(T,k);
    [U{k},S,V]=svd(C.data);
    
    DimS=size(S);
    Sigma=diag(S(1:min(DimS),1:min(DimS)));
    Sigma=Sigma.^2;
    FVE=cumsum(Sigma)/sum(Sigma);
    NPC(k)=find(FVE>=FVED,1);
    U{k}=U{k}(:,1:NPC(k));
    Tucker=ttm(Tucker,U{k}',k);
end
end