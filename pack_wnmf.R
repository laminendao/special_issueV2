#Calcul_Hp retourne les tableau dijonctifs des classifications obtenue sur chaque tableau
#entrée: l'ensemble des partitions P ==option partitions
#l'ensemble des tableaux M ==option tableau
#k est le nombre de classe
#--------------------Chargement des packages-----------------------
library(quadprog)
library(FactoMineR)
library(NMF)
library(Matrix)
#-------------------------------------------Données pour tester----------------
source("script/pack_statis.R")
#------------------------------------------------------------------------------

#Cette fonction retourne une liste de matrices d'indicatrices
# Entrées: liste de tableau M, ou liste de de partition P

Calcul_Hp=function(Tab, option,nb){
  H_comp=vector(mode = "list",length = length(Tab))
	  if(option=="P"){
	    for (i in seq_along(Tab)){
	      
	      H_comp[[i]]=FactoMineR::tab.disjonctif(as.data.frame(Tab[[i]]))
	      
	    }
	   }
      
	  else if(option=="M"){
	    for (i in seq_along(Tab)){
	    H_comp[[i]]=FactoMineR::tab.disjonctif(kmeans(as.data.frame(Tab[[i]]),nb)$cluster)
	    }
	  }
return(H_comp)
}




#Calcul de l'nsemble des M matrice de connectivité

# Elle retourne une liste de M


Calcul_allM=function(Tab, option,nb){
  #H_comp=vector(mode = "list",length = length(Tab))
  all_M=vector(mode = "list",length = length(Tab))
  
  if(option=="P"){
    for (i in seq_along(Tab)){
      H_ <- FactoMineR::tab.disjonctif(as.data.frame(Tab[[i]]))#Calcul de H
      all_M[[i]] <- H_%*%t(H_)
      #H_comp[[i]]=FactoMineR::tab.disjonctif(as.data.frame(Tab[[i]]))
    }
  }
  
  else if(option=="M"){
    for (i in seq_along(Tab)){
      H_ <-FactoMineR::tab.disjonctif(kmeans(as.data.frame(Tab[[i]]),nb)$cluster)
      all_M[[i]] <- H_%*%t(H_)
    }
  }
  return(all_M)
}


#res <- Calcul_Hp(classes,option="P")

#----------------------------------------------------------------------------------

#Poid_init retourne les poids initialisés de façon uniforme
#Elle prend en arguments une liste de partitions ou de tableaux

#A supprimer

Poid_init=function(part){
  nb_part=length(part) #Nombre de partitions
  
  W_ini=vector(mode = "double",length = nb_part)#initialisation des poids à un vecteur null
  
  for (i in 1:nb_part){
    W_ini[i]=1/nb_part #vecteur contenant les poids uniformes 1/T
  }
  return(W_ini)
}

#--------------------------------------------------------------

#Calcul_A retourne une matrice carrée de dimensions (nb_partition,nb_partition)
#Elle prend en argument une liste de matrice d'indicatrice H_, calcul les matrices de connectivité
# et retourne 

#A est une matrice carré de dimensions n,n

Calcul_A=function(H_){
  nb_part=length(H_)
  H_ <- lapply(H_, as.matrix)
  A=matrix(0,nb_part,nb_part) #Initialiser A(nb_partition,nb_partition)
  for (i in 1:nb_part){
    for(j in 1:nb_part){
      A[i,j]=Trace((H_[[i]]%*%t(H_[[i]]))%*%(H_[[j]]%*%t(H_[[j]])))
    }
  }
  return(A)
}

#Calcul_A(res)

#--------------------------------------------------------------------------------

#Cette fonction retourne une matrice moyenne pondérée
#Elle prend en argument Une liste de matrice et un vecteur poids

Moy_pond=function(list_mat,W_){
  nb_part=length(list_mat)
  list_mat=lapply(list_mat, as.matrix)
  
  M_avg=matrix(0, dim(list_mat[[1]])[1],dim(list_mat[[1]])[1])#Initialisation d'une matrice vide
  
  for (i in 1:nb_part){
    M_avg=M_avg+W_[i]*list_mat[[i]]%*%t(list_mat[[i]])
  }
  return(M_avg)
}

#w_ <- Poid_init(res)
#Moy_pond(res,W_)
#----------------------------------------------------------------

#Source::Vincent Audigier
#Dans cette fonction j'ai juste utilisé la partie sur la factorisation par NMF de la matrice Htilde

nmf_<-function(Mtilde,nb.clust,threshold=10^(-5),printflag=TRUE,nstart=100,iter.max=50){

  #initialisation
  #Mtilde <- H_tilde
  #H_tilde <- Mtilde
  res.kmeans<-try(kmeans(Mtilde,centers = nb.clust,nstart = nstart,iter.max=iter.max))
  if("try-error"%in% class(res.kmeans)){
    res.kmeans<-sample(seq(nb.clust),size = ncol(Mtilde),replace=TRUE)
  }else{
    res.kmeans<-res.kmeans$cluster
  }
  
  H<-FactoMineR::tab.disjonctif(res.kmeans)
  Htilde<-H%*%diag(diag(1/sqrt(crossprod(H))),nb.clust,nb.clust) #equivalent de G dans Matlab
  S<-crossprod(H)
  continue<-TRUE
  critsave<-sqrt(sum(diag(crossprod(Mtilde-Htilde%*%tcrossprod(S,Htilde)))))
  comp<-1
  while(continue){
    if(printflag){cat(comp,"...")}
    #Htilde update
    MHS <- Mtilde%*%Htilde%*%S
    multHtilde<-sqrt(MHS/(tcrossprod(Htilde)%*%MHS))
    multHtilde[is.nan(multHtilde)]<-0
    Htilde<-Htilde*multHtilde
    #S update
    cpHtilde <- crossprod(Htilde)
    multS<-sqrt((crossprod(Htilde,Mtilde)%*%Htilde)/(cpHtilde%*%S%*%cpHtilde))
    multS[is.nan(multS)]<-0
    multS[is.infinite(multS)]<-0
    S<-S*multS
    
    critsave<-c(critsave,sqrt(sum(diag(crossprod(Mtilde-Htilde%*%tcrossprod(S,Htilde))))))
    comp<-comp+1
    diffcrit<-critsave[comp-1]-critsave[comp]
    continue<-(diffcrit>=threshold)
  }
  if(printflag){cat("done \n")}
  
  res<-list(Htilde=Htilde,S=S,crit=critsave,Mtilde=Mtilde,cluster=apply(Htilde,1,which.max))
  return(res)
}
# Est équivalente à la fonction TriNMF de Matlab: au critère d'arret près
#------------------------------------------------------

# Cette fonction retourne le vecteur b utile pour calculer les poids par la

# suite
# Entree: H: c'est la matrice que nous avons trouvé par NMF
#         M: C'est la structure contenant toutes la matrices de connectivité des partitions 
# Sortie: bvect: c'est un vecteur de longueur T= nombre de partition=nb matrice de connectivité  
# 
# nb_part est le nombre des partitions

bVect=function(H,M){
  nb_part=length(M)
  M=lapply(M, as.matrix)
  H <- as.matrix(H)
  bvect=vector(mode="double",length=nb_part)
  for (i in 1:nb_part){
    bvect[i]=Trace((t(H)%*%(M[[i]])%*%H))
  }
  return(bvect)
}

#----------------------------Algorithme W_NMF------------------------------

#Programme qui prend en entrée un ensemble de tableau ou cluster et donne en sortie un cluster
#Avec les poids contributifs

consensus_wnmf <- function(M,nb_cluster,option="M",maxiter=30){
  nb_class <- nb_cluster
  all_H=Calcul_Hp(M,option=option,nb_class)#Calcul des matrices de connectivité
  all_M=Calcul_allM(M,option = option,nb_class)
  C=as.matrix(nearPD(Calcul_A(all_H))$mat)            #Calcul de la matrice A
  W=Poid_init(all_H)
  r=rep(1,length(all_H))
  A <- matrix(1,1,length(all_H))
  A <- rbind(A, r, diag(length(all_H)),-diag(length(all_H)))
  
  #----------------------------Algorithme W_NMF------------------------------
  nb_iter=0
  critere_=TRUE
  epsilon=10^(-3)
  critere_err=0
  
  while(critere_){
    #elle représente la matrice M_tilde calculer en fonction des poids et des matrices de connectivité
    M_avg=Moy_pond(all_H,W)
    #fixer les poids et résoudre H_tild
    #Wi <- W
    H=nmf_(M_avg,nb_class)$Htilde
    #Fixer H_tild et resoudre les poids
    f <- c(1, 0.00000001, rep(0,length(all_H)),rep(-1,length(all_H)))
    W <- solve.QP(Dmat=C, dvec = (-1)*(bVect(H,all_M)), Amat=t(A), bvec=f, meq=1)$solution #Optimisation
    
    if(nb_iter==0){
      critere_err <- Trace(M_avg%*%M_avg)-2*Trace(t(H)%*%M_avg%*%H)+
        Trace(crossprod(H%*%t(H),t(H%*%t(H))))
      taux_err <- critere_err
      nb_iter=nb_iter+1
      critere_ <- as.logical(taux_err>epsilon)
    }
    else{
      critere_err_i <- critere_err
      critere_err <- Trace(M_avg%*%M_avg)-2*Trace(t(H)%*%M_avg%*%H)+
        Trace(crossprod(H%*%t(H),t(H%*%t(H))))
      
      ecart_ <- critere_err_i-critere_err
      taux_err <- append(taux_err,ecart_)
      nb_iter=nb_iter+1
      #critere_ <- as.logical(sum(abs(Wi-W))>epsilon)
      critere_ <- as.logical(ecart_>epsilon)
    }
  }

cluster_ <- apply(H, 1,which.max)
res_wnmf=list(cluster=cluster_,W=W,H=H,nb_iter=nb_iter,taux_erreur=taux_err)
return(res_wnmf)
}


# 
# liste_dat <- list(don,don,don,don)
# list_tableau <- res.imp[1:100]
# 
#res_ <- consensus_wnmf(list_30part,3,option = "P")
# res_$cluster
# #length(res_$taux_erreur)# 
# A=res_$H
# sqrt(sum(diag(crossprod(A))))
# 
# View(res.imp[1:10])
