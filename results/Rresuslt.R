# Faire une acp
library(tidyverse)
library(FactoMineR)
library(ade4)
library(factoextra)

df = formated_result%>%rownames_to_column("nom_ligne")
#df_pca = all_result[,c(4, 5, 6, 7, 8, 9,10, 11, 12, 16, 17, 18, 13)]
#df_pca = all_result[,c(4, 5, 6, 7, 8, 9,10, 11, 12, 16, 17, 18, 13)]
#df_pca_test = filter(df_pca, explainer!="saliency")
#df_pca_test = df_pca_test[,1:8]



df_pca = formated_result[, c(4:5, 7:9, 13)]
row.names(df_pca) = unique(formated_result$nom_ligne)
res.pca=PCA(df_pca, scale.unit=TRUE, graph=T)


pca <- prcomp(df_pca, 
              scale = TRUE)
fviz_pca_ind(res.pca)

da = df_pca

summary(df_pca)
# acp=dudi.pca(df_pca,scannf=F,center=T,scale=T)
acp = pca
summary(acp)
fviz_eig(res.pca)
fviz_eig(acp, addlabels = TRUE, ylim = c(0, 50))

# corection de saporta-kaiser
n=nrow(da)
p=ncol(da)
q=1+2*sqrt((p-1)/(n-1))
q


fviz_pca_ind(acp,
             col.ind = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", 
                               "#FC4E07"),repel = TRUE)
