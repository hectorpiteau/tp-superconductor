library("FactoMineR")
library(PCAmixdata)
load("/cloud/project/Data/Projets/superconductivity_data_train.rda")
# Affichage des données
data<-data_train
print(data,digits=4)

# Calcul de la moyenne et de l’écart type des variables
mean <- apply(data,2,mean)
std <- apply(data,2,sd) #standard deviation
stat <- rbind(mean,std)
# Affichage
print(stat,digits=4)

# Création des données centrées ...
datanorm <- sweep(data,2,mean,"-")
# ... et réduites
datanorm <- sweep(datanorm,2,std,"/")
# Affichage des données centrées - réduites
print(datanorm,digits=4)

# Analyse en composantes principales sur les données d’origine
# (scale.unit=FALSE)
res <- PCA(data,graph=FALSE,scale.unit=FALSE)
# Figure individus
plot(res,choix="ind",cex=1.5,title="")
# Figure variables
plot(res,choix="var",cex=1.5,title="")


# Analyse en composantes principales sur les données centrées-réduites
# (par défaut: scale.unit=TRUE)
resnorm <- PCA(data,graph=FALSE)
# Figure individus
plot(resnorm,choix="ind",cex=1.5,title="")
# Figure variables
plot(resnorm,choix="var",cex=1.5,title="")


# Inertie (variance) des composantes principales
resnorm$eig
barplot(resnorm$eig[,1])

# Contribution des individus
resnorm$ind$contrib
# Projection des variables
resnorm$var$cos2
resnorm$var$cos2[,1]+resnorm$var$cos2[,2]+resnorm$var$cos2[,3]+resnorm$var$cos2[,4]+resnorm$var$cos2[,5]

