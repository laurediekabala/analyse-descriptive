import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Pca_Analysis:
     def __init__(self,col_index:str):
          """Initialisation des paramètres
        
        Parameters:
        -----------
        col_index : str
            Nom de la colonne à utiliser comme index
          """
          self.col_index=col_index
          self.eigval =None
          self.p =None
          self.n =None
          self.sc=None
          self.data_cpa=None
          self.y=None
          self.df=None
          self.pca=None
          self.coord= None
          self.corvar=None
          self.corr=None
          self.coorv=None
     def transformation(self,data)  :
          """Transformation des données pour l'ACP
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Données à transformer
          """
          data.index = data[self.col_index]
          self.y= data[self.col_index]
          data=data.select_dtypes(include=np.number)
          self.df=data
          self.sc=StandardScaler()
          self.data_cpa=self.sc.fit_transform(self.df.copy())
          self.p = self.data_cpa.shape[1]
          self.n=  self.data_cpa.shape[0]
          self.pca=PCA()
          self.coord=self.pca.fit_transform(self.data_cpa)
     def composante(self,data)  :
         """ici nous allons afficher les composantes principales""" 
         self.transformation(data)
         print(self.pca.explained_variance_ ) 
         # correction sur les valeurs propres
         self.eigval =(self.n-1)/self.n*self.pca.explained_variance_
         print(self.eigval)
         df=pd.DataFrame({"dimension":[f"dim{str(i+1)}"for i in range(self.p)],"eigval":self.eigval,"variance":np.round(self.pca.explained_variance_ratio_*100),"cumul":np.round(np.cumsum(self.pca.explained_variance_ratio_)*100)})
         print(df)  
     def plot_scree(self, figsize=(12, 6)):
        """
        Graphique des valeurs propres (scree plot) avec variance expliquée
        """   
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        # Graphique 1: Valeurs propres
        ax1.bar(np.arange(1,self.p+1), self.eigval, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.plot(np.arange(1,self.p+1), self.eigval, 'ro-', linewidth=2, markersize=8)
        ax1.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Seuil Kaiser (λ=1)')
        ax1.set_xlabel('Composante Principale', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Valeur Propre', fontsize=11, fontweight='bold')
        ax1.set_title('Éboulis des Valeurs Propres', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3) 

        # Graphique 2: Variance expliquée
        variance_pct = np.round(self.pca.explained_variance_ratio_*100)
        cumulative_variance_pct = np.round(np.cumsum(self.pca.explained_variance_ratio_)*100)
        
        ax2.bar(np.arange(1,self.p+1), variance_pct, color='lightcoral', 
               alpha=0.7, label='Individuelle', edgecolor='black')
        ax2.plot(np.arange(1,self.p+1), cumulative_variance_pct, 'go-', 
                linewidth=2, markersize=8, label='Cumulée')
        ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Seuil 80%')
        ax2.set_xlabel('Composante Principale', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Variance Expliquée (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Variance Expliquée par Composante', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 105)
        
        plt.tight_layout()
        return fig
     def qualite_represent_indiv(self) :
        # qualités de représention des individus
        # pour répresenter les individus sur les facteurs retenus ,il faut  qu'on trouve d'abord 
        # les carrés  des distances des individus à l'origine.Ces carrés correspondent à la contribution pour la formation des axes.
        di = np.sum(self.coord**2,axis=1)
        cos2=self.coord**2
        for j in range(self.p) :
            cos2[:,j]=cos2[:,j]/di
        qualit_indiv=pd.DataFrame({"individus":self.df.index,"cos1":cos2[:,0],"cos2":cos2[:,1]})
        print(qualit_indiv)
        # contribution des individus aux axes(ctr)
        ctr = self.coord**2
        for j in range(self.p):
            ctr[:,j]=ctr[:,j]/(self.n*self.eigval[j])
        cont_indi=pd.DataFrame({"individus":self.df.index,"ctr1":ctr[:,0],"ctr2":ctr[:,1]}) 
        return qualit_indiv,cont_indi
     def qualite_represent_var(self) :
         # qualité de répresentation des variables
         # Représention des variables sur les axes
        # pour obtenir la corrélation entre variables et facteurs nous devons multiplier  la matrice  var x fac par la racine carré des valeurs propres
        eigsqrt= np.sqrt(self.eigval)
        # correltion des vars avec les axes
        self.corvar=np.zeros((self.p,self.p))
        for k in range(self.p) :
            # matrice de correlation vars x facteurs 
            self.corvar[:,k]=self.pca.components_[k,:]*eigsqrt[k]
        cos2var= self.corvar**2
        qualit_var=pd.DataFrame({"var":self.df.columns,"cos1":cos2var[:,0],"cos2":cos2var[:,1]})
        # vérification de la theorie
        print(np.sum(cos2var,axis=1))
        #contribution des variables
        ctrv=cos2var
        for k in range(self.p):
            ctrv[:,k] = ctrv[:,k]/self.eigval[k]
        contr_var=pd.DataFrame({"var":self.df.columns,"cos1":np.round(ctrv[:,0]*100),"cos2":np.round(ctrv[:,1]*100)})   
        print(np.sum(ctrv,axis=0))
        return qualit_var,contr_var
     def cercle_correlation(self)  :
        #cercle de correlation
        plt.figure(figsize=(10,10))
        for i in range(self.p) :
            plt.arrow(0,0,self.corvar[i,0],self.corvar[i,1],color="k",alpha=0.4,head_width=0.02)
            plt.text(self.corvar[i,0]*1.0,self.corvar[i,1]*1.05,self.df.columns[i],color='red',ha="center",va="center")
        circle =plt.Circle((0,0),1,fill=False,color="blue",linestyle="--")
        plt.gca().add_artist(circle)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.axhline(0,color="blue")
        plt.axvline(0,color="blue")
        plt.xlabel("pc1")
        plt.ylabel("pc2")
     def carte_indiv(self)  :
        #carte des individus
        plt.figure(figsize=(10,10))
        for i in range(self.n) :
            plt.text(self.coord[i,0],self.coord[i,1],self.df.index[i],color='green',ha="center",va="center",fontsize=10)
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.axhline(y=0, color='red', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='red', linestyle='-', linewidth=0.5)
        plt.xlabel("pc1")
        plt.ylabel("pc2")
     def biplot(self)  :
        # biplot
        plt.figure(figsize=(10,10))
        # individus
        for i in range(self.n) :
            plt.text(self.coord[i,0],self.coord[i,1],self.df.index[i],color='green',ha="center",va="center")
        # variables
        for i in range(self.p) :
            plt.arrow(0,0,self.corvar[i,0],self.corvar[i,1],color="k",alpha=0.4,head_width=0.02)
            plt.text(self.corvar[i,0]*1.0,self.corvar[i,1]*1.05,self.df.columns[i],color='red',ha="center",va="center")
        circle =plt.Circle((0,0),1,fill=False,color="blue",linestyle="--")
        plt.gca().add_artist(circle)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.axhline(0,color="blue")
        plt.axvline(0,color="blue")
        plt.xlabel("pc1")
        plt.ylabel("pc2") 
     def grap_cercle_correlation_var_ind(self)  :
   
        """
        Affiche le cercle de corrélation combiné avec la carte des individus colorés selon la variable qualitative (ex : continent).
        Montre également les barycentres des classes.
        """
    # Création du DataFrame de coordonnées
        self.corr = pd.DataFrame({
            "f1": self.coord[:, 0],
            "f2": self.coord[:, 1]
        }).reset_index(drop=True)
        # Conversion de la variable qualitative
        y_series = pd.Series(self.y).reset_index(drop=True)
        if len(y_series) != len(self.corr):
            raise ValueError(f"Dimension incompatible : y ({len(y_series)}) vs coord ({len(self.corr)})")
        self.corr["var"] = y_series
        # Moyenne conditionnelle (barycentres)
        self.coorv = pd.pivot_table(self.corr, index="var", values=["f1", "f2"], aggfunc="mean")
        coorx = self.coorv.values
        # --- Création du graphique ---
        plt.figure(figsize=(10, 10))
           # 1️⃣ Affichage des individus colorés selon la variable qualitative
        sns.scatterplot(
        x="f1", y="f2", hue="var",
        data=self.corr, alpha=0.7, s=60, palette="tab10", edgecolor="none"
    )
         # 2️⃣ Affichage des barycentres
        for i in range(self.coorv.shape[0]):
          plt.scatter(coorx[i, 0], coorx[i, 1], color='black', s=80, marker='X')
          plt.annotate(self.coorv.index[i],
                     (coorx[i, 0], coorx[i, 1]),
                     fontsize=11, fontweight='bold', color='black', xytext=(5, 5),
                     textcoords="offset points")
         # 3️⃣ Cercle de corrélation (variables)
        #for i in range(self.p):
          #plt.arrow(0, 0, self.corvar[i, 0], self.corvar[i, 1],
                  #color="gray", alpha=0.5, head_width=0.03, length_includes_head=True)
          #plt.text(self.corvar[i, 0]*1.1, self.corvar[i, 1]*1.1,
                 #self.df.columns[i], color='red', ha="center", va="center")
        # 4️⃣ Cercle unité
        #circle = plt.Circle((0, 0), 1, fill=False, color="blue", linestyle="--")
        #plt.gca().add_artist(circle)

        # 5️⃣ Ajustement automatique des axes
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        # 6️⃣ Esthétique
        plt.axhline(0, color="blue", linestyle="--", linewidth=0.8)
        plt.axvline(0, color="blue", linestyle="--", linewidth=0.8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Cercle de corrélation et individus colorés par continent", fontsize=14)
        plt.legend(title="Continent", loc="best", fontsize=9)
        plt.grid(alpha=0.2)
        plt.show()

     
 





          


          
        
    