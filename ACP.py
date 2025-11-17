import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ACPAnalyzer:
    """
    Classe pour l'Analyse en Composantes Principales (ACP)
    Compatible avec les résultats de R (FactoMineR, factoextra)
    """
    
    def __init__(self, n_components=None, scale=True):
        """
        Paramètres:
        -----------
        n_components : int, optionnel
            Nombre de composantes principales à retenir
        scale : bool, default=True
            Si True, standardise les variables (centre-réduit)
        """
        self.n_components = n_components
        self.scale = scale
        self.mean_ = None
        self.std_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None
        self.components_ = None
        self.transformed_data_ = None
        self.loadings_ = None
        self.squared_cosines_ = None
        self.contributions_var_ = None
        self.contributions_ind_ = None
        self.quality_representation_var_ = None
        self.quality_representation_ind_ = None
        self.feature_names_ = None
        self.individual_names_ = None
        
    def fit_transform(self, X, feature_names=None, individual_names=None):
        """
        Ajuste l'ACP et transforme les données
        
        Paramètres:
        -----------
        X : array-like, shape (n_samples, n_features)
            Données d'entrée
        feature_names : list, optionnel
            Noms des variables
        individual_names : list, optionnel
            Noms des individus
        
        Retourne:
        ---------
        transformed_data : array, shape (n_samples, n_components)
            Données transformées (coordonnées des individus)
        """
        # Conversion en array numpy
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            if individual_names is None:
                individual_names = X.index.tolist()
            X = X.values
        
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        
        # Stockage des noms
        self.feature_names_ = feature_names if feature_names else [f"Var{i+1}" for i in range(n_features)]
        self.individual_names_ = individual_names if individual_names else [f"Ind{i+1}" for i in range(n_samples)]
        
        # Centrage des données (obligatoire pour ACP)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Standardisation si demandé (ACP normée comme dans R)
        if self.scale:
            self.std_ = np.std(X, axis=0, ddof=1)  # ddof=1 comme dans R
            # Éviter division par zéro
            self.std_[self.std_ == 0] = 1
            X_centered = X_centered / self.std_
        else:
            self.std_ = np.ones(n_features)
        
        # Matrice de corrélation (si scale=True) ou covariance (si scale=False)
        # Calcul comme dans R avec ddof=1
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        # Décomposition en valeurs propres et vecteurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Tri par ordre décroissant (comme dans R)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Détermination du nombre de composantes
        if self.n_components is None:
            self.n_components = n_features
        else:
            self.n_components = min(self.n_components, n_features)
        
        # Stockage des résultats
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        
        # Composantes principales (vecteurs propres)
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Transformation des données (coordonnées des individus)
        self.transformed_data_ = np.dot(X_centered, self.components_)
        
        # Loadings (corrélations variables-composantes)
        # Comme dans R: correlation entre variable et composante
        self.loadings_ = self.components_ * np.sqrt(self.eigenvalues_[:self.n_components])
        
        # Calcul des contributions et qualités de représentation
        self._compute_contributions_and_quality()
        
        return self.transformed_data_
    
    def _compute_contributions_and_quality(self):
        """
        Calcule les contributions et qualités de représentation
        Formules conformes à R (FactoMineR)
        """
        n_samples = self.transformed_data_.shape[0]
        
        # --- CONTRIBUTIONS DES VARIABLES ---
        # Contribution = (coordonnée²) / valeur propre * 100
        self.contributions_var_ = (self.loadings_ ** 2) / self.eigenvalues_[:self.n_components] * 100
        
        # --- QUALITÉ DE REPRÉSENTATION DES VARIABLES (cos²) ---
        # cos² = (coordonnée²) / somme des coordonnées² sur toutes les composantes
        squared_loadings = self.loadings_ ** 2
        sum_squared_loadings = np.sum(squared_loadings, axis=1, keepdims=True)
        self.quality_representation_var_ = squared_loadings / sum_squared_loadings
        
        # --- CONTRIBUTIONS DES INDIVIDUS ---
        # Contribution = (coordonnée²) / (n * valeur propre) * 100
        self.contributions_ind_ = (self.transformed_data_ ** 2) / (n_samples * self.eigenvalues_[:self.n_components]) * 100
        
        # --- QUALITÉ DE REPRÉSENTATION DES INDIVIDUS (cos²) ---
        # Distance à l'origine pour chaque individu
        squared_coords = self.transformed_data_ ** 2
        total_inertia_ind = np.sum(squared_coords, axis=1, keepdims=True)
        # Éviter division par zéro
        total_inertia_ind[total_inertia_ind == 0] = 1
        self.quality_representation_ind_ = squared_coords / total_inertia_ind
    
    def get_top_contributors_variables(self, component=0, n_top=10):
        """
        Retourne les variables qui contribuent le plus à une composante
        
        Paramètres:
        -----------
        component : int
            Indice de la composante (0 pour PC1, 1 pour PC2, etc.)
        n_top : int
            Nombre de variables à retourner
        
        Retourne:
        ---------
        DataFrame avec les contributions triées
        """
        contrib = self.contributions_var_[:, component]
        idx = np.argsort(contrib)[::-1][:n_top]
        
        return pd.DataFrame({
            'Variable': [self.feature_names_[i] for i in idx],
            'Contribution (%)': contrib[idx],
            'Qualité (cos²)': self.quality_representation_var_[idx, component]
        })
    
    def get_top_contributors_individuals(self, component=0, n_top=10):
        """
        Retourne les individus qui contribuent le plus à une composante
        
        Paramètres:
        -----------
        component : int
            Indice de la composante
        n_top : int
            Nombre d'individus à retourner
        
        Retourne:
        ---------
        DataFrame avec les contributions triées
        """
        contrib = self.contributions_ind_[:, component]
        idx = np.argsort(contrib)[::-1][:n_top]
        
        return pd.DataFrame({
            'Individu': [self.individual_names_[i] for i in idx],
            'Contribution (%)': contrib[idx],
            'Qualité (cos²)': self.quality_representation_ind_[idx, component]
        })
    
    def plot_correlation_circle(self, components=(0, 1), figsize=(12, 10), n_labels=None):
        """
        Cercle des corrélations (graphique des variables)
        
        Paramètres:
        -----------
        components : tuple
            Indices des composantes à afficher (PC1, PC2)
        figsize : tuple
            Taille de la figure
        n_labels : int, optionnel
            Nombre de labels à afficher (les mieux représentés)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extraire les loadings pour les composantes sélectionnées
        pc1, pc2 = components
        x = self.loadings_[:, pc1]
        y = self.loadings_[:, pc2]
        
        # Qualité de représentation (moyenne des cos² sur les deux axes)
        quality = (self.quality_representation_var_[:, pc1] + 
                  self.quality_representation_var_[:, pc2]) / 2
        
        # Cercle de corrélation
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
        ax.add_artist(circle)
        
        # Déterminer quelles variables afficher
        if n_labels is not None:
            idx_to_label = np.argsort(quality)[::-1][:n_labels]
        else:
            idx_to_label = range(len(x))
        
        # Tracer les flèches et labels
        for i in range(len(x)):
            # Couleur basée sur la qualité de représentation
            color = plt.cm.RdYlGn(quality[i])
            
            # Flèche
            ax.arrow(0, 0, x[i]*0.95, y[i]*0.95, 
                    head_width=0.03, head_length=0.03, 
                    fc=color, ec=color, alpha=0.7, linewidth=2)
            
            # Label seulement pour les variables sélectionnées
            if i in idx_to_label:
                ax.text(x[i]*1.05, y[i]*1.05, self.feature_names_[i],
                       fontsize=9, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor=color, alpha=0.8))
        
        # Axes
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        # Labels et titre
        var_explained_1 = self.explained_variance_ratio_[pc1] * 100
        var_explained_2 = self.explained_variance_ratio_[pc2] * 100
        ax.set_xlabel(f'PC{pc1+1} ({var_explained_1:.2f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc2+1} ({var_explained_2:.2f}%)', fontsize=12, fontweight='bold')
        ax.set_title('Cercle des Corrélations - Contribution des Variables', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Limites
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        
        # Grille
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar pour la qualité
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Qualité de représentation (cos²)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_individuals_map(self, components=(0, 1), figsize=(12, 10), 
                            n_labels=20, color_by_contribution=True):
        """
        Carte des individus
        
        Paramètres:
        -----------
        components : tuple
            Indices des composantes à afficher
        figsize : tuple
            Taille de la figure
        n_labels : int
            Nombre de labels à afficher
        color_by_contribution : bool
            Colorer par contribution totale
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        pc1, pc2 = components
        x = self.transformed_data_[:, pc1]
        y = self.transformed_data_[:, pc2]
        
        # Contribution totale sur les deux axes
        contrib_total = (self.contributions_ind_[:, pc1] + 
                        self.contributions_ind_[:, pc2])
        
        # Normaliser pour la couleur
        if color_by_contribution:
            colors = contrib_total
            cmap = 'viridis'
        else:
            colors = 'steelblue'
            cmap = None
        
        # Scatter plot
        scatter = ax.scatter(x, y, c=colors, s=100, alpha=0.6, 
                           cmap=cmap, edgecolors='black', linewidth=0.5)
        
        # Labels pour les top contributeurs
        idx_to_label = np.argsort(contrib_total)[::-1][:n_labels]
        for i in idx_to_label:
            ax.annotate(self.individual_names_[i], (x[i], y[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                               alpha=0.5, edgecolor='gray'))
        
        # Axes
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        
        # Labels et titre
        var_explained_1 = self.explained_variance_ratio_[pc1] * 100
        var_explained_2 = self.explained_variance_ratio_[pc2] * 100
        ax.set_xlabel(f'PC{pc1+1} ({var_explained_1:.2f}%)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC{pc2+1} ({var_explained_2:.2f}%)', fontsize=12, fontweight='bold')
        ax.set_title('Carte des Individus - Contribution à la Formation des Axes', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Grille
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Colorbar
        if color_by_contribution:
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Contribution totale (%)', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_scree(self, figsize=(12, 6)):
        """
        Graphique des valeurs propres (scree plot) avec variance expliquée
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        n_comp = len(self.eigenvalues_)
        x = np.arange(1, n_comp + 1)
        
        # Graphique 1: Valeurs propres
        ax1.bar(x, self.eigenvalues_, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.plot(x, self.eigenvalues_, 'ro-', linewidth=2, markersize=8)
        ax1.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Seuil Kaiser (λ=1)')
        ax1.set_xlabel('Composante Principale', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Valeur Propre', fontsize=11, fontweight='bold')
        ax1.set_title('Éboulis des Valeurs Propres', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Variance expliquée
        variance_pct = self.explained_variance_ratio_ * 100
        cumulative_variance_pct = self.cumulative_variance_ratio_ * 100
        
        ax2.bar(x[:self.n_components], variance_pct, color='lightcoral', 
               alpha=0.7, label='Individuelle', edgecolor='black')
        ax2.plot(x[:self.n_components], cumulative_variance_pct, 'go-', 
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
    
    def plot_contribution_barplot(self, component=0, n_top=15, entity='variables', figsize=(12, 8)):
        """
        Graphique en barres des contributions
        
        Paramètres:
        -----------
        component : int
            Indice de la composante
        n_top : int
            Nombre d'éléments à afficher
        entity : str
            'variables' ou 'individuals'
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if entity == 'variables':
            contrib = self.contributions_var_[:, component]
            names = self.feature_names_
            title = f'Contribution des Variables à PC{component+1}'
        else:
            contrib = self.contributions_ind_[:, component]
            names = self.individual_names_
            title = f'Contribution des Individus à PC{component+1}'
        
        # Tri et sélection des top contributeurs
        idx = np.argsort(contrib)[::-1][:n_top]
        contrib_sorted = contrib[idx]
        names_sorted = [names[i] for i in idx]
        
        # Moyenne de contribution (ligne de référence)
        mean_contrib = 100 / len(contrib)
        
        # Graphique
        colors = ['crimson' if c > mean_contrib else 'steelblue' for c in contrib_sorted]
        bars = ax.barh(range(len(contrib_sorted)), contrib_sorted, color=colors, 
                       alpha=0.7, edgecolor='black')
        
        ax.set_yticks(range(len(contrib_sorted)))
        ax.set_yticklabels(names_sorted, fontsize=10)
        ax.set_xlabel('Contribution (%)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.axvline(x=mean_contrib, color='red', linestyle='--', linewidth=2, 
                  label=f'Contribution moyenne ({mean_contrib:.2f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Inverser l'ordre pour avoir le plus grand en haut
        ax.invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def get_summary(self):
        """
        Retourne un résumé de l'ACP
        """
        summary = {
            'n_components': self.n_components,
            'eigenvalues': self.eigenvalues_[:self.n_components],
            'variance_explained': self.explained_variance_ratio_ * 100,
            'cumulative_variance': self.cumulative_variance_ratio_ * 100
        }
        
        df = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(self.n_components)],
            'Valeur Propre': self.eigenvalues_[:self.n_components],
            'Variance (%)': self.explained_variance_ratio_ * 100,
            'Variance Cumulée (%)': self.cumulative_variance_ratio_ * 100
        })
        
        return df


# =====================================================
# FONCTION PRINCIPALE D'ANALYSE ACP
# =====================================================

def analyser_acp_complet(data, n_components=None, scale=True, 
                        feature_names=None, individual_names=None,
                        n_top_var=15, n_top_ind=20, n_labels_circle=None,
                        output_plots=True):
    """
    Fonction complète d'analyse ACP avec tous les graphiques et résultats
    
    Paramètres:
    -----------
    data : DataFrame ou array
        Données à analyser
    n_components : int, optionnel
        Nombre de composantes à retenir
    scale : bool, default=True
        Standardisation des variables (ACP normée)
    feature_names : list, optionnel
        Noms des variables
    individual_names : list, optionnel
        Noms des individus
    n_top_var : int
        Nombre de variables top contributrices à afficher
    n_top_ind : int
        Nombre d'individus top contributeurs à afficher
    n_labels_circle : int, optionnel
        Nombre de labels dans le cercle des corrélations
    output_plots : bool
        Afficher les graphiques
    
    Retourne:
    ---------
    dict contenant:
        - 'model': Modèle ACP ajusté
        - 'summary': Résumé de la variance expliquée
        - 'top_variables_pc1': Top contributeurs PC1
        - 'top_variables_pc2': Top contributeurs PC2
        - 'top_individuals_pc1': Top contributeurs PC1
        - 'top_individuals_pc2': Top contributeurs PC2
        - 'transformed_data': Données transformées
    """
    print("="*70)
    print("ANALYSE EN COMPOSANTES PRINCIPALES (ACP)")
    print("Implémentation compatible avec R (FactoMineR)")
    print("="*70)
    
    # Initialisation et ajustement du modèle
    acp = ACPAnalyzer(n_components=n_components, scale=scale)
    transformed_data = acp.fit_transform(data, feature_names, individual_names)
    
    # Affichage du résumé
    print("\n📊 RÉSUMÉ DE LA VARIANCE EXPLIQUÉE")
    print("-" * 70)
    summary = acp.get_summary()
    print(summary.to_string(index=False))
    print("-" * 70)
    
    # Top contributeurs pour PC1
    print("\n🔝 TOP VARIABLES CONTRIBUTRICES À PC1")
    print("-" * 70)
    top_var_pc1 = acp.get_top_contributors_variables(component=0, n_top=n_top_var)
    print(top_var_pc1.to_string(index=False))
    
    # Top contributeurs pour PC2
    print("\n🔝 TOP VARIABLES CONTRIBUTRICES À PC2")
    print("-" * 70)
    top_var_pc2 = acp.get_top_contributors_variables(component=1, n_top=n_top_var)
    print(top_var_pc2.to_string(index=False))
    
    # Top individus pour PC1
    print("\n👥 TOP INDIVIDUS CONTRIBUTEURS À PC1")
    print("-" * 70)
    top_ind_pc1 = acp.get_top_contributors_individuals(component=0, n_top=n_top_ind)
    print(top_ind_pc1.to_string(index=False))
    
    # Top individus pour PC2
    print("\n👥 TOP INDIVIDUS CONTRIBUTEURS À PC2")
    print("-" * 70)
    top_ind_pc2 = acp.get_top_contributors_individuals(component=1, n_top=n_top_ind)
    print(top_ind_pc2.to_string(index=False))
    
    # Génération des graphiques
    if output_plots:
        print("\n📈 GÉNÉRATION DES GRAPHIQUES...")
        
        # 1. Éboulis des valeurs propres
        acp.plot_scree()
        plt.show()
        
        # 2. Cercle des corrélations
        acp.plot_correlation_circle(n_labels=n_labels_circle)
        plt.show()
        
        # 3. Carte des individus
        acp.plot_individuals_map(n_labels=n_top_ind)
        plt.show()
        
        # 4. Contributions des variables PC1
        acp.plot_contribution_barplot(component=0, n_top=n_top_var, entity='variables')
        plt.show()
        
        # 5. Contributions des variables PC2
        acp.plot_contribution_barplot(component=1, n_top=n_top_var, entity='variables')
        plt.show()
        
        # 6. Contributions des individus PC1
        acp.plot_contribution_barplot(component=0, n_top=n_top_ind, entity='individuals')
        plt.show()
        
        # 7. Contributions des individus PC2
        acp.plot_contribution_barplot(component=1, n_top=n_top_ind, entity='individuals')
        plt.show()
    
    print("\n✅ ANALYSE COMPLÉTÉE AVEC SUCCÈS!")
    print("="*70)
    
    # Retour des résultats
    return {
        'model': acp,
        'summary': summary,
        'top_variables_pc1': top_var_pc1,
        'top_variables_pc2': top_var_pc2,
        'top_individuals_pc1': top_ind_pc1,
        'top_individuals_pc2': top_ind_pc2,
        'transformed_data': pd.DataFrame(
            transformed_data,
            columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])],
            index=acp.individual_names_
        )
    }


# =====================================================
# EXEMPLE D'UTILISATION AVEC DONNÉES ÉCONOMIQUES
# =====================================================


    
  