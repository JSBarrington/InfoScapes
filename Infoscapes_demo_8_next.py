#!/usr/bin/env python3
"""
Infoscapes Studio - Final Working Version
Information-theoretic Species Delimitation with multiple analytical lenses.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances, mutual_info_score
from scipy.spatial.distance import pdist, jensenshannon
from scipy.stats import gaussian_kde, entropy
from scipy.ndimage import label, laplace
from scipy.interpolate import griddata
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from numpy import linalg
import warnings
from io import StringIO
import itertools

from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import squareform
from scipy import stats as sps


warnings.filterwarnings('ignore')

# Optional dependency checks
OPTIONAL_LIBS = {}
try:
    import umap
    OPTIONAL_LIBS['UMAP'] = True
except ImportError:
    OPTIONAL_LIBS['UMAP'] = False

try:
    from Bio import SeqIO
    from Bio.Align import MultipleSeqAlignment
    from Bio.Phylo.TreeConstruction import DistanceCalculator
    OPTIONAL_LIBS['Bio'] = True
except ImportError:
    OPTIONAL_LIBS['Bio'] = False

try:
    import pacmap
    OPTIONAL_LIBS['PaCMAP'] = True
except ImportError:
    OPTIONAL_LIBS['PaCMAP'] = False
    
try:
    import networkx as nx
    OPTIONAL_LIBS['networkx'] = True
except ImportError:
    OPTIONAL_LIBS['networkx'] = False

try:
    import phate
    OPTIONAL_LIBS['PHATE'] = True
except ImportError:
    OPTIONAL_LIBS['PHATE'] = False

try:
    import kmapper as km
    OPTIONAL_LIBS['kmapper'] = True
except ImportError:
    OPTIONAL_LIBS['kmapper'] = False

try:
    import graphtools as gt
    OPTIONAL_LIBS['graphtools'] = True
except ImportError:
    OPTIONAL_LIBS['graphtools'] = False


# Page Configuration
st.set_page_config(page_title="Infoscapes Studio", page_icon="🗻", layout="wide", initial_sidebar_state="expanded")

# Dark theme styling
st.markdown("""<style>.stApp { background-color: #282c34; color: #f0f2f6; } h1, h2, h3, h4, h5, h6 { color: #f0f2f6 !important; } h1 { border-bottom: 3px solid #305D57 !important; padding-bottom: 0.5rem !important; } [data-testid="stAlert"] { border-radius: 12px; padding: 1rem; } [data-testid="stAlert"][kind="success"] { background-color: #305D57; color: #f0f2f6; } [data-testid="stAlert"][kind="info"] { background-color: #1E3C43; color: #f0f2f6; } [data-testid="stSidebar"] { background-color: #1E3C43; } .stButton > button { background-color: #305D57 !important; color: white !important; border-radius: 12px !important; font-weight: 500 !important; border: 1px solid #4a9c8c !important;} .stButton > button:hover { background-color: #4a9c8c !important; } [data-testid="metric-container"] { background: rgba(48, 93, 87, 0.2); border-radius: 12px; padding: 1rem; border: 1px solid #305D57; } [data-testid="metric-container"] [data-testid="metric-label"] { color: #D6E5C6 !important; } [data-testid="metric-container"] [data-testid="metric-value"] { color: #f0f2f6 !important; } .stTabs [data-baseweb="tab-list"] { background: rgba(30, 60, 67, 0.8); border-radius: 12px; padding: 0.25rem; } .stTabs [aria-selected="true"] { background: #305D57 !important; color: white !important; border-radius: 8px; } .stMarkdown, .stText, p, span, div { color: #f0f2f6 !important; }</style>""", unsafe_allow_html=True)


class InfoscapeAnalyzer:
    """Core analysis engine for Infoscapes Studio."""
    
    def compute_embedding(self, data, method='PCA', n_components=2, **kwargs):
        scaler = StandardScaler()
        is_dist_matrix = hasattr(data, 'shape') and len(data.shape) == 2 and data.shape[0] == data.shape[1] and np.allclose(data, data.T)
        data_scaled = data if is_dist_matrix else scaler.fit_transform(data)
        
        methods = {
            'PCA': lambda: PCA(n_components=n_components, random_state=42).fit_transform(data_scaled),
            't-SNE': lambda: TSNE(n_components=n_components, perplexity=kwargs.get('perplexity', min(30, len(data) - 1)), random_state=42, init='pca', learning_rate='auto').fit_transform(data_scaled),
            'MDS': lambda: MDS(n_components=n_components, random_state=42, dissimilarity='euclidean', normalized_stress='auto').fit_transform(data_scaled),
            'PCoA (MDS)': lambda: self.compute_pcoa(data if is_dist_matrix else pdist(data_scaled), n_components),
            'Laplacian Eigenmaps': lambda: SpectralEmbedding(n_components=n_components, n_neighbors=kwargs.get('n_neighbors', min(15, len(data) - 1)), random_state=42).fit_transform(data_scaled)
        }
        
        if OPTIONAL_LIBS['UMAP']: methods['UMAP'] = lambda: umap.UMAP(n_components=n_components, n_neighbors=kwargs.get('n_neighbors', min(15, len(data) - 1)), random_state=42).fit_transform(data_scaled)
        if OPTIONAL_LIBS['PaCMAP']: methods['PaCMAP'] = lambda: pacmap.PaCMAP(n_components=n_components, n_neighbors=kwargs.get('n_neighbors', min(15, len(data) - 1)), random_state=42).fit_transform(data_scaled)
        if OPTIONAL_LIBS['networkx'] and n_components == 2: methods['k-NN Graph'] = lambda: self.compute_knn_graph(data_scaled, **kwargs)
        if OPTIONAL_LIBS['PHATE']: methods['PHATE'] = lambda: phate.PHATE(n_components=n_components, n_jobs=-1, random_state=42).fit_transform(data_scaled)
        if OPTIONAL_LIBS['graphtools']: methods['Diffusion Map'] = lambda: gt.Graph(data_scaled, use_pygsp=True).to_pygsp().U[:, 1:n_components+1]

        if method in methods:
            return methods[method]()
        else:
            st.error(f"Method {method} is not available or library not installed.")
            return None
            
    def compute_pcoa(self, distance_matrix, n_components):
        return MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, normalized_stress='auto').fit_transform(distance_matrix)
    
    def compute_knn_graph(self, data, k=10, layout='spring'):
        from sklearn.neighbors import kneighbors_graph
        A = kneighbors_graph(data, k, mode='distance', include_self=False)
        G = nx.from_scipy_sparse_array(A)
        pos = nx.spring_layout(G, seed=42) if layout == 'spring' else nx.spectral_layout(G)
        return np.array([pos[i] for i in range(len(data))])
    
    def detect_basins(self, landscape, threshold=0.3):
        binary = landscape < (landscape.min() + (landscape.max() - landscape.min()) * threshold)
        labeled_basins, _ = label(binary)
        return labeled_basins

    def compute_consensus_clustering(self, data, n_runs=10, method='DBSCAN', **kwargs):
        n_samples = len(data)
        coassoc = np.zeros((n_samples, n_samples))
        all_labels = []
        for run in range(n_runs):
            data_noisy = data + np.random.normal(0, 0.01, data.shape)
            if method == 'DBSCAN': labels = DBSCAN(eps=kwargs.get('eps', np.percentile(pdist(data_noisy), 15))).fit_predict(data_noisy)
            else: labels = KMeans(n_clusters=kwargs.get('n_clusters_base', 4), random_state=run, n_init='auto').fit_predict(data_noisy)
            all_labels.append(labels)
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] != -1 and labels[i] == labels[j]:
                        coassoc[i, j] += 1; coassoc[j, i] += 1
        coassoc /= n_runs
        np.fill_diagonal(coassoc, 1.0)
        final_clustering = AgglomerativeClustering(n_clusters=kwargs.get('n_clusters_final', 4), metric='precomputed', linkage='average')
        return final_clustering.fit_predict(1 - coassoc), coassoc, all_labels

    def compute_variation_information(self, all_labels):
        n_runs = len(all_labels)
        if n_runs < 2: return 0.0
        vi_matrix = np.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                mask = (all_labels[i] >= 0) & (all_labels[j] >= 0)
                l1, l2 = all_labels[i][mask], all_labels[j][mask]
                if len(l1) == 0: vi = 0.0
                else: vi = (entropy(np.bincount(l1)) + entropy(np.bincount(l2)) - 2 * mutual_info_score(l1, l2))
                vi_matrix[i, j] = vi_matrix[j, i] = vi
        return np.mean(vi_matrix[np.triu_indices(n_runs, 1)])
    
    def compute_jsd_matrix(self, data, labels):
        unique_labels = sorted([l for l in np.unique(labels) if l != -1])
        n_clusters = len(unique_labels)
        if n_clusters < 2: return pd.DataFrame()
        jsd_matrix = pd.DataFrame(np.nan, index=unique_labels, columns=unique_labels)
        xx, yy = np.mgrid[data[:,0].min():data[:,0].max():100j, data[:,1].min():data[:,1].max():100j]
        grid_points = np.vstack([xx.ravel(), yy.ravel()])
        cluster_kdes = {}
        for label in unique_labels:
            cluster_points = data[labels == label]
            if len(cluster_points) <= data.shape[1]:
                st.warning(f"Cluster {label} has too few points ({len(cluster_points)}) to compute a stable density estimate. It will be excluded from JSD calculations.")
                continue
            try:
                kde = gaussian_kde(cluster_points.T)
                density = np.reshape(kde(grid_points).T, xx.shape)
                cluster_kdes[label] = density.ravel() / density.sum()
            except linalg.LinAlgError:
                st.warning(f"Could not compute density for Cluster {label} due to a singular matrix (points may be collinear). It will be excluded from JSD calculations.")
                continue
        for i in range(n_clusters):
            for j in range(i, n_clusters):
                l1, l2 = unique_labels[i], unique_labels[j]
                if l1 in cluster_kdes and l2 in cluster_kdes:
                    jsd_matrix.loc[l1, l2] = jsd_matrix.loc[l2, l1] = jensenshannon(cluster_kdes[l1], cluster_kdes[l2], base=2)
        return jsd_matrix
        
    def identify_unstable_samples(self, coassoc_matrix, labels):
        instability = []
        for i in range(len(labels)):
            if labels[i] != -1 and len(np.where(labels == labels[i])[0]) > 0:
                cluster_coassoc = np.mean(coassoc_matrix[i, np.where(labels == labels[i])[0]])
                instability.append(1.0 - cluster_coassoc)
            else: instability.append(1.0)
        return np.array(instability)

    def compute_knn_density(self, points_2d, grid_points, k):
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(points_2d)
        distances, _ = nn.kneighbors(grid_points)
        k_dist = distances[:, -1]
        density = 1.0 / (k_dist**2 + 1e-9) 
        return density

    def calculate_barrier_height(self, landscape, xx, yy, points, labels, cluster1_id, cluster2_id):
        c1_points = points[labels == cluster1_id]
        c2_points = points[labels == cluster2_id]
        if len(c1_points) == 0 or len(c2_points) == 0: return None, None, None
        c1_centroid = c1_points.mean(axis=0)
        c2_centroid = c2_points.mean(axis=0)
        path = np.array([np.linspace(c1_centroid[i], c2_centroid[i], 200) for i in range(2)]).T
        path_heights = griddata(np.vstack([xx.ravel(), yy.ravel()]).T, landscape.ravel(), path, method='linear')
        path_heights = np.nan_to_num(path_heights, nan=np.max(landscape))
        floor1 = griddata(np.vstack([xx.ravel(), yy.ravel()]).T, landscape.ravel(), c1_points, method='linear').min()
        floor2 = griddata(np.vstack([xx.ravel(), yy.ravel()]).T, landscape.ravel(), c2_points, method='linear').min()
        saddle_point_height = path_heights.max()
        highest_floor = max(floor1, floor2)
        barrier_height = saddle_point_height - highest_floor
        return barrier_height, saddle_point_height, highest_floor

    # <<< NEW FEATURE: METHODS FOR AUTOMATED PERSISTENCE RUN >>>
    def compute_coclustering_matrix(self, partitions):
        n_samples = partitions.shape[1]
        co_cluster = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Count how many times sample i and j are in the same cluster
                count = np.sum(partitions[:, i] == partitions[:, j])
                co_cluster[i, j] = co_cluster[j, i] = count
        np.fill_diagonal(co_cluster, len(partitions))
        return co_cluster / len(partitions)

    def compute_partition_agreement(self, partitions):
        vi_scores = []
        for p1, p2 in itertools.combinations(partitions, 2):
            mask = (p1 >= 0) & (p2 >= 0)
            l1, l2 = p1[mask], p2[mask]
            if len(l1) == 0: vi = 0.0
            else: vi = (entropy(np.bincount(l1)) + entropy(np.bincount(l2)) - 2 * mutual_info_score(l1, l2))
            vi_scores.append(vi)
        return np.mean(vi_scores) if vi_scores else 0

def create_3d_landscape_with_wall_projections(xx, yy, zz, points=None, labels=None, plot_options=None, sequence_ids=None, selected_ids=None):
    fig = go.Figure()
    plot_options = plot_options or {}
    if plot_options.get('show_surface', True):
        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale=[[0, '#305D57'], [0.5, '#D6E5C6'], [1, '#FFFFFF']], opacity=0.9, showscale=True, colorbar=dict(title=dict(text="U(x)", font=dict(color='white')), tickfont=dict(color='white')), name="Landscape", hoverinfo='none', contours=dict(x=dict(show=plot_options.get('project_y', True), usecolormap=True, highlightcolor="#f0f2f6", project=dict(x=True)), y=dict(show=plot_options.get('project_x', True), usecolormap=True, highlightcolor="#f0f2f6", project=dict(y=True)), z=dict(show=plot_options.get('project_z', True), usecolormap=True, highlightcolor="#f0f2f6", project=dict(z=True)))))
    if points is not None:
        z_values = griddata(np.vstack([xx.ravel(), yy.ravel()]).T, zz.ravel(), points, method='linear', fill_value=np.min(zz))
        z_values = np.nan_to_num(z_values, nan=np.min(zz)) + (zz.max()-zz.min())*0.02
        hover_text = np.array(sequence_ids if sequence_ids else [f'Sample {i}' for i in range(len(points))])
        point_sizes = plot_options.get('point_size', 8)
        base_marker_data = {'x': points[:, 0], 'y': points[:, 1], 'z': z_values, 'mode': 'markers', 'hovertext': hover_text, 'hoverinfo': 'text'}
        fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=np.full(len(points), zz.min()), mode='markers', marker=dict(color='rgba(0,0,0,0)'), hovertext=hover_text, hoverinfo='text', showlegend=False))
        if labels is not None and len(np.unique(labels)) > 1:
            unique_labels = sorted([l for l in np.unique(labels) if l != -1])
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            for i, label in enumerate(unique_labels):
                mask = np.array(labels == label)
                if np.any(mask):
                    trace_data = {k: v[mask] if isinstance(v, np.ndarray) else v for k, v in base_marker_data.items()}
                    fig.add_trace(go.Scatter3d(**trace_data, name=f'Cluster {label}', marker=dict(size=point_sizes, color=colors[i % len(colors)])))
            if np.any(labels == -1):
                mask = np.array(labels == -1)
                if np.any(mask):
                    trace_data = {k: v[mask] if isinstance(v, np.ndarray) else v for k, v in base_marker_data.items()}
                    fig.add_trace(go.Scatter3d(**trace_data, name='Noise', marker=dict(size=2, color='white', opacity=0.6)))
        else:
            fig.add_trace(go.Scatter3d(**base_marker_data, name='Data Points', marker=dict(size=point_sizes, color='#1f77b4')))
        if selected_ids and sequence_ids:
            sel_indices = [list(sequence_ids).index(sid) for sid in selected_ids if sid in sequence_ids]
            if sel_indices:
                fig.add_trace(go.Scatter3d(x=points[sel_indices, 0], y=points[sel_indices, 1], z=z_values[sel_indices], text=hover_text[sel_indices], mode='text', textposition='middle right', hoverinfo='none', name='Labels', showlegend=False, textfont=dict(color='#FFFFFF')))
    fig.update_layout(title_text="🗻 Information Landscape", height=800, scene=dict(xaxis_title='Dimension 1', yaxis_title='Dimension 2', zaxis_title='Information Height', bgcolor='#282c34', xaxis=dict(showspikes=True, spikecolor="grey", title=dict(font=dict(color='white')), tickfont=dict(color='white')), yaxis=dict(showspikes=True, spikecolor="grey", title=dict(font=dict(color='white')), tickfont=dict(color='white')), zaxis=dict(showspikes=True, spikecolor="grey", title=dict(font=dict(color='white')), tickfont=dict(color='white'))), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(30,60,67,0.8)', font=dict(color="#f0f2f6")), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def clear_analysis_state():
    keys_to_clear = ['embedded', 'landscape', 'xx', 'yy', 'cluster_labels', 'coassoc_matrix', 'avg_vi', 'jsd_matrix', 'instability_scores', 'persistence_results']
    for key in keys_to_clear:
        if key in st.session_state: del st.session_state[key]

def create_tda_graph(graph, labels, sequence_ids, instability_scores, node_size_metric, color_map):
    if not OPTIONAL_LIBS['networkx']:
        st.error("NetworkX library not found. Please install it to use this feature.")
        return go.Figure()
    nodes = graph['nodes']
    links = graph['links']
    G = nx.Graph()
    for node_id, members in nodes.items():
        G.add_node(node_id, members=members)
    for link in links:
        G.add_edge(link[0], link[1])
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_size, node_text, node_color = [], [], [], [], []
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x); node_y.append(y)
        members = G.nodes[node_id].get('members', [])
        if node_size_metric == "Number of Samples":
            node_size.append(len(members) * 2)
            size_val = len(members)
        else:
            member_instability = instability_scores[members] if len(members) > 0 else [0]
            avg_instability = np.mean(member_instability)
            node_size.append(avg_instability * 50 + 5)
            size_val = f"{avg_instability:.3f}"
        member_labels = labels[members]
        dominant_label = -1
        if len(member_labels) > 0:
            dominant_label = pd.Series(member_labels).mode()[0]
        node_color.append(dominant_label)
        member_ids = np.array(sequence_ids)[members]
        hover_text = (f"<b>Node {node_id}</b><br>"
                      f"{node_size_metric}: {size_val}<br>"
                      f"Dominant Cluster: {dominant_label if dominant_label != -1 else 'N/A'}<br>"
                      f"Members ({len(members)}): {', '.join(member_ids[:5])}{'...' if len(member_ids) > 5 else ''}")
        node_text.append(hover_text)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                            marker=dict(showscale=True, colorscale=color_map, size=node_size, color=node_color,
                                        colorbar=dict(thickness=15, title=dict(text='Dominant Cluster ID', side='right'), xanchor='left'),
                                        line_width=2))
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=dict(text='🌐 Topological Data Analysis (Mapper) Graph', font=dict(size=16)),
                                     showlegend=False, hovermode='closest',
                                     margin=dict(b=20,l=5,r=5,t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     template='plotly_dark'))
    return fig
    

# === Correlation Explorer Helpers ===
import scipy
from scipy.cluster.hierarchy import linkage, leaves_list

def _to_numeric_df(df_like, ids=None):
    import numpy as np, pandas as pd
    if isinstance(df_like, np.ndarray):
        X = pd.DataFrame(df_like, index=ids)
    else:
        X = pd.DataFrame(df_like)
    X = X.select_dtypes(include=[float, int])
    return X

def compute_correlation_df(X_df, method: str = "pearson", shrinkage=False):
    import numpy as np, pandas as pd
    if shrinkage:
        lw = LedoitWolf().fit(X_df.values)
        C = lw.covariance_
        d = np.sqrt(np.diag(C))
        corr = C / np.outer(d, d)
        corr = np.clip(corr, -1, 1)
        return pd.DataFrame(corr, index=X_df.columns, columns=X_df.columns)
    if method in {"pearson", "spearman", "kendall"}:
        return X_df.corr(method=method)
    raise ValueError(f"Unknown correlation method: {method}")

def cluster_reorder(corr_df):
    import numpy as np
    dist = 1.0 - np.abs(corr_df.values)
    dist = np.clip(dist, 0, 1)
    condensed = scipy.spatial.distance.squareform((dist + dist.T)/2, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return corr_df.iloc[order, :].iloc[:, order]

def plot_corr_heatmap(corr_df, title="Correlation Heatmap", mask_upper=False):
    import numpy as np
    M = corr_df.values.copy()
    if mask_upper:
        iu = np.triu_indices_from(M, k=1)
        M[iu] = np.nan
    fig = go.Figure(
        data=go.Heatmap(
            z=M, x=corr_df.columns, y=corr_df.index,
            colorscale="RdBu", zmin=-1, zmax=1, reversescale=True, hoverongaps=False
        )
    )
    fig.update_layout(title=title, template='plotly_dark', height=700)
    return fig

def mantel_test(A, B, n_perm: int = 999, method: str = "pearson", random_state: int = 42):
    """A and B must be square distance-like matrices with same shape."""
    import numpy as np
    rng = np.random.default_rng(random_state)
    assert A.shape == B.shape and A.shape[0] == A.shape[1]
    a = scipy.spatial.distance.squareform(A, checks=False)
    b = scipy.spatial.distance.squareform(B, checks=False)
    if method == "pearson":
        r_obs = np.corrcoef(a, b)[0, 1]
    else:
        r_obs = scipy.stats.spearmanr(a, b).correlation
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(B.shape[0])
        bp = scipy.spatial.distance.squareform(B[p][:, p], checks=False)
        if method == "pearson":
            r = np.corrcoef(a, bp)[0, 1]
        else:
            r = scipy.stats.spearmanr(a, bp).correlation
        if abs(r) >= abs(r_obs):
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return r_obs, pval

# === Cluster Distribution Fit Helpers ===
def _safe_minmax01(x):
    import numpy as np
    x = np.asarray(x, float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn <= 0:
        return np.zeros_like(x)
    y = (x - mn) / (mx - mn)
    # Avoid exact 0/1 which break beta logpdf
    eps = 1e-6
    y = np.clip(y, eps, 1 - eps)
    return y

def _loglik_norm(x, params):
    mu, sigma = params
    return np.sum(sps.norm.logpdf(x, loc=mu, scale=max(sigma, 1e-9)))

def _loglik_beta(x01, params):
    a, b = params
    return np.sum(sps.beta.logpdf(x01, a=max(a,1e-6), b=max(b,1e-6)))

def _loglik_gamma(xpos, params):
    a, loc, scale = params
    return np.sum(sps.gamma.logpdf(xpos, a=max(a,1e-6), loc=loc, scale=max(scale,1e-9)))

def _loglik_poisson(k, lam):
    return np.sum(sps.poisson.logpmf(k, mu=max(lam, 1e-9)))

def best_fit_distributions_1d(x, var_type="continuous"):
    """
    Returns a list of dicts: {'name','aic','bic','params'} sorted by AIC ascending.
    var_type: 'continuous', 'bounded01', or 'count'
    """
    import numpy as np
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    fits = []

    if len(x) < 5:
        return fits

    if var_type == "continuous":
        # Normal
        p_norm = sps.norm.fit(x)  # (mu, sigma)
        ll_norm = _loglik_norm(x, p_norm)
        k_norm = 2
        fits.append({"name":"Normal", "aic": 2*k_norm - 2*ll_norm, "bic": k_norm*np.log(len(x)) - 2*ll_norm, "params": p_norm})

        # Gamma (x must be positive)
        x_pos = x[x > 0]
        if len(x_pos) >= 5:
            p_gamma = sps.gamma.fit(x_pos)  # (a, loc, scale)
            ll_gamma = _loglik_gamma(x_pos, p_gamma)
            k_gamma = 3
            # Penalize for subsetting if many zeros: adjust n to len(x_pos)
            fits.append({"name":"Gamma", "aic": 2*k_gamma - 2*ll_gamma, "bic": k_gamma*np.log(len(x_pos)) - 2*ll_gamma, "params": p_gamma})

    if var_type == "bounded01":
        x01 = _safe_minmax01(x)
        p_beta = sps.beta.fit(x01, floc=0, fscale=1)  # force support [0,1]
        ll_beta = _loglik_beta(x01, p_beta[:2])
        k_beta = 2
        fits.append({"name":"Beta", "aic": 2*k_beta - 2*ll_beta, "bic": k_beta*np.log(len(x01)) - 2*ll_beta, "params": p_beta})

        # Also try Normal as a sanity baseline on transformed variable
        p_norm = sps.norm.fit(x01)
        ll_norm = _loglik_norm(x01, p_norm)
        k_norm = 2
        fits.append({"name":"Normal(on 0-1)", "aic": 2*k_norm - 2*ll_norm, "bic": k_norm*np.log(len(x01)) - 2*ll_norm, "params": p_norm})

    if var_type == "count":
        k = np.rint(x).astype(int)
        if np.all(k >= 0):
            lam = np.mean(k)
            ll_pois = _loglik_poisson(k, lam)
            k_p = 1
            fits.append({"name":"Poisson", "aic": 2*k_p - 2*ll_pois, "bic": k_p*np.log(len(k)) - 2*ll_pois, "params": (lam,)})

            # Also compare Normal on counts (overdispersed alternative baseline)
            p_norm = sps.norm.fit(k)
            ll_norm = _loglik_norm(k, p_norm)
            k_norm = 2
            fits.append({"name":"Normal(counts)", "aic": 2*k_norm - 2*ll_norm, "bic": k_norm*np.log(len(k)) - 2*ll_norm, "params": p_norm})

    fits.sort(key=lambda d: d["aic"])
    return fits

def plot_fit_overlay(x, fit_list, var_type="continuous", title="Distribution Fit"):
    import numpy as np
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    fig = go.Figure()
    # Histogram
    histnorm = 'probability density' if var_type != "count" else ''
    fig.add_trace(go.Histogram(x=x, nbinsx=30, histnorm=histnorm, name="Data", opacity=0.6))

    xs = np.linspace(np.min(x), np.max(x), 300) if var_type != "count" else np.arange(np.min(x), np.max(x)+1)
    for fit in fit_list[:3]:  # overlay up to 3 best
        name = fit["name"]
        if var_type == "count" and "Poisson" in name:
            ys = sps.poisson.pmf(xs, fit["params"][0])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f"{name}"))
        elif "Gamma" in name:
            a, loc, scale = fit["params"]
            ys = sps.gamma.pdf(xs, a, loc=loc, scale=scale)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f"{name}"))
        elif "Beta" in name:
            xs01 = _safe_minmax01(xs)
            a, b, loc, scale = fit["params"]
            ys = sps.beta.pdf(xs01, a, b, loc=0, scale=1)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f"{name}"))
        else:
            mu, sigma = fit["params"][:2]
            ys = sps.norm.pdf(xs, mu, sigma)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=f"{name}"))
    fig.update_layout(title=title, template='plotly_dark', height=500)
    return fig


def render_sidebar():
    st.sidebar.header("Analysis Pipeline")
    with st.sidebar.expander("Optional Library Status", expanded=False):
        for lib, available in OPTIONAL_LIBS.items():
            st.markdown(f"- **{lib}**: {'✅ Available' if available else '❌ Not Found'}")
    with st.sidebar.expander("Step 1: Data Input", expanded=True):
        data_sources = ["Demo Data", "Upload CSV"] + (["Upload FASTA"] if OPTIONAL_LIBS['Bio'] else [])
        data_source = st.radio("Source:", data_sources)
        if data_source == "Demo Data":
            if st.button("Generate New Demo Data", use_container_width=True):
                st.session_state.data, _ = make_blobs(n_samples=200, n_features=10, centers=4, cluster_std=1.5, random_state=np.random.randint(0,1000))
                st.session_state.sequence_ids = [f"Demo_{i}" for i in range(200)]
                st.session_state.feature_names = [f"F{j}" for j in range(st.session_state.data.shape[1])]
                clear_analysis_state(); st.rerun()
        elif data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Choose CSV (for feature data or pre-computed distance matrices)", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file, header=None if st.checkbox("CSV has no header row", key='csv_header') else 'infer', index_col=0 if st.checkbox("First column is sample ID", key='csv_index') else None)
                if df.shape[0] == df.shape[1] and np.allclose(df.values, df.values.T):
                     st.info("Square matrix detected. Treating as a distance matrix.")
                     sample_ids = list(df.columns.astype(str))
                     if st.button("Load Distance Matrix", use_container_width=True):
                        st.session_state.data = df.values
                        st.session_state.feature_names = None
                        st.session_state.sequence_ids = sample_ids
                        clear_analysis_state(); st.rerun()
                else:
                    selected_cols = st.multiselect("Select numerical features:", df.select_dtypes(include=np.number).columns.tolist(), default=df.select_dtypes(include=np.number).columns.tolist())
                    if st.button("Load Feature Data", use_container_width=True) and selected_cols:
                        st.session_state.data = df[selected_cols].values
                        st.session_state.feature_names = list(selected_cols)
                        st.session_state.sequence_ids = df.index.astype(str).tolist()
                        clear_analysis_state(); st.rerun()
        elif data_source == "Upload FASTA":
            uploaded_fasta = st.file_uploader("Choose FASTA", type=["fasta", "fa", "fas"])
            if uploaded_fasta:
                fasta_method = st.radio(
                    "FASTA Processing:",
                    ["Alignment-Free (k-mer)", "Alignment-Based (Identity)", "Alignment-Based (Hamming)"]
                )
                k_mer_size = 5
                if fasta_method == "Alignment-Free (k-mer)":
                    k_mer_size = st.slider("k-mer size", 3, 8, 5)
                if st.button("Process FASTA", use_container_width=True):
                    with st.spinner("Processing sequences..."):
                        sequences = list(SeqIO.parse(StringIO(uploaded_fasta.getvalue().decode("utf-8")), "fasta"))
                        st.session_state.sequence_ids = [s.id for s in sequences]
                        if fasta_method == "Alignment-Free (k-mer)":
                            k = k_mer_size
                            all_kmers = {str(s.seq[i:i+k]) for s in sequences for i in range(len(s.seq)-k+1)}
                            k_list = list(all_kmers)
                            k_map = {kmer: i for i, kmer in enumerate(k_list)}
                            fm = np.zeros((len(sequences), len(k_list)))
                            for i, s in enumerate(sequences):
                                for j in range(len(s.seq)-k+1):
                                    kmer = str(s.seq[j:j+k])
                                    if kmer in k_map:
                                        fm[i, k_map[kmer]] += 1
                            from sklearn.metrics.pairwise import cosine_distances
                            st.session_state.data = cosine_distances(fm)
                            st.session_state.feature_names = None
                        elif fasta_method == "Alignment-Based (Hamming)":
                            seq_matrix = np.array([list(rec.seq) for rec in sequences])
                            seq_matrix_numeric = np.array(seq_matrix).view(np.int32)
                            st.session_state.data = pairwise_distances(seq_matrix_numeric, metric='hamming')
                            st.session_state.feature_names = None
                        else:
                            from Bio.SeqRecord import SeqRecord
                            aln = MultipleSeqAlignment([SeqRecord(s.seq, id=s.id) for s in sequences])
                            st.session_state.data = np.array(DistanceCalculator('identity').get_distance(aln))
                            st.session_state.feature_names = None
                        clear_analysis_state(); st.rerun()
    if 'data' not in st.session_state or st.session_state.data is None:
        st.sidebar.warning("Load data to begin the analysis.")
        return None, None

    with st.sidebar.expander("Step 2: Interactive Analysis", expanded=True):
        methods = ["PCA", "t-SNE", "MDS", "PCoA (MDS)", "Laplacian Eigenmaps"]
        for lib in ['UMAP', 'PaCMAP', 'PHATE', 'Diffusion Map' if OPTIONAL_LIBS.get('graphtools') else None]:
             if lib: methods.append(lib)
        if st.session_state.get('n_components', 2) == 2 and OPTIONAL_LIBS['networkx']: 
            methods.append("k-NN Graph")
        st.session_state.embedding_method = st.selectbox("Method:", sorted(methods))
        st.session_state.n_components = st.radio("Embedding Dimensions:", [2, 3], index=0, horizontal=True)
        st.session_state.embedding_kwargs = {}
        if st.session_state.embedding_method in ["t-SNE", "UMAP", "PaCMAP", "Laplacian Eigenmaps"]:
            n_samples = len(st.session_state.data)
            max_neighbors = max(2, n_samples - 1)
            st.session_state.embedding_kwargs['n_neighbors'] = st.slider("Neighbors", 2, max_neighbors, min(15, max_neighbors-1))
        st.session_state.landscape_maker = st.radio("Landscape Type:", ["KDE", "k-NN Density", "GMM"])
        st.session_state.transform_type = st.radio("Landscape Z-Axis:", ["Information Height", "Potential Energy", "Probability Density", "Gradient (Cliffs)", "Laplacian of Density"])
        if st.session_state.landscape_maker == "KDE": st.session_state.kde_bandwidth = st.slider("KDE Bandwidth", 0.01, 2.0, 0.5, step=0.01)
        elif st.session_state.landscape_maker == "k-NN Density":
            n_samples = len(st.session_state.data); max_k = max(2, n_samples - 1)
            st.session_state.knn_k = st.slider("k-Neighbors for Density", 2, max_k, min(15, max_k-1))
        elif st.session_state.landscape_maker == "GMM":
            st.session_state.gmm_n_components = st.slider("GMM Components (Clusters)", 2, 20, 4)
        st.session_state.landscape_extent = st.slider("Landscape Extent", 0.1, 2.0, 0.2)
        st.session_state.cluster_method = st.selectbox("Clustering Algorithm:", ["None", "DBSCAN", "KMeans", "Consensus"])
        st.session_state.cluster_kwargs = {}
        if st.session_state.cluster_method == "KMeans": st.session_state.cluster_kwargs['n_clusters'] = st.slider("Number of clusters (K)", 2, 15, 4)
        if st.session_state.cluster_method == "Consensus":
            st.session_state.cluster_kwargs['method'] = st.selectbox("Base method", ["DBSCAN", "KMeans"])
            if st.session_state.cluster_kwargs['method'] == "KMeans": st.session_state.cluster_kwargs['n_clusters_base'] = st.slider("Base K for each run", 2, 15, 4)
            st.session_state.cluster_kwargs['n_clusters_final'] = st.slider("Final K", 2, 15, 4)
            st.session_state.cluster_kwargs['n_runs'] = st.slider("Consensus runs", 5, 50, 10)
        
        run_interactive = st.button("🚀 Run Interactive Analysis", type="primary", use_container_width=True)
    
    # <<< NEW FEATURE: AUTOMATED PERSISTENCE RUN UI >>>
    with st.sidebar.expander("Step 3: Automated Persistence Run", expanded=False):
        st.info("This runs multiple DR methods and compares the resulting clusters to test for robustness.")
        all_methods = ["PCA", "t-SNE", "MDS", "Laplacian Eigenmaps"] + [lib for lib in ['UMAP', 'PaCMAP', 'PHATE', 'Diffusion Map'] if OPTIONAL_LIBS.get(lib.split(' ')[0].lower(), False)]
        st.session_state.persistence_methods = st.multiselect("Select DR Methods to Compare:", all_methods, default=all_methods)
        st.session_state.persistence_k = st.slider("Number of Clusters (K) for Persistence Run", 2, 15, 4)
        run_persistence = st.button("🤖 Run Persistence Analysis", use_container_width=True)

    if OPTIONAL_LIBS['kmapper']:
        with st.sidebar.expander("TDA (Mapper) Parameters", expanded=False):
            st.info("These settings only affect the 'TDA (Mapper) Graph' tab.")
            st.session_state.tda_lens_type = st.selectbox("TDA Lens/Filter:", ["PCA_1", "PCA_2", "L2-norm"])
            st.session_state.tda_intervals = st.slider("Number of Intervals", 2, 50, 10)
            st.session_state.tda_overlap = st.slider("Percent Overlap (%)", 10, 90, 50)
            st.session_state.tda_dbscan_eps = st.slider("TDA DBSCAN eps", 0.1, 5.0, 1.5, step=0.1)
            
    return run_interactive, run_persistence


def render_main_content(analyzer):
    # If a cross-method persistence run has been requested and computed, show that first
    if 'persistence_results' in st.session_state:
        st.success("Persistence Analysis Complete!")
        results = st.session_state.persistence_results
        st.metric("Mean Pairwise Variation of Information", f"{results['mean_vi']:.4f}", help="Average disagreement between all pairs of partitions. Lower is better.")
        st.markdown("### Co-Clustering Heatmap")
        st.info("This heatmap shows the frequency that any two samples were clustered together across all selected DR methods.")
        fig = go.Figure(data=go.Heatmap(z=results['cocluster_matrix'], x=results['sample_ids'], y=results['sample_ids'], colorscale='viridis'))
        fig.update_layout(title="Cross-Method Co-clustering Frequency", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Partitions from each Method")
        st.dataframe(results['partitions_df'])
        return

    if 'embedded' not in st.session_state:
        st.info("👈 Use the sidebar to load data and configure an analysis.")
        return

    st.success("Interactive Analysis Complete!")
    tabs_to_show = ["🗻 Landscape", "🧬 Coherence", "↔️ Separation", "🔄 Persistence", "📈 Correlations", "📊 Distributions"]
    if OPTIONAL_LIBS['kmapper']: tabs_to_show.append("🌐 TDA (Mapper) Graph")
    if st.session_state.embedded.shape[1] == 3: 
        tabs_to_show.insert(1, "🎲 3D Embedding Scatter")
    active_tabs = st.tabs(tabs_to_show)
    tab_offset = 1 if st.session_state.embedded.shape[1] == 3 else 0

    # --- Landscape Tab ---
    with active_tabs[0]:
        c1, c2, c3, c4, c5 = st.columns(5)
        plot_options = {'show_surface': c1.checkbox("Surface", value=True), 'project_z': c2.checkbox("Floor (XY)", value=True), 'project_x': c3.checkbox("X Wall (YZ)", value=True), 'project_y': c4.checkbox("Y Wall (XZ)", value=True)}
        plot_options['point_size'] = c5.slider("Point Size", 2, 20, 8)
        selected_ids = st.multiselect("Highlight Sample Labels:", options=st.session_state.get('sequence_ids', []))
        st.plotly_chart(
            create_3d_landscape_with_wall_projections(
                st.session_state.xx, st.session_state.yy, st.session_state.landscape, 
                st.session_state.embedded[:,:2], st.session_state.get('cluster_labels'),
                plot_options=plot_options, sequence_ids=st.session_state.get('sequence_ids'),
                selected_ids=selected_ids
            ), use_container_width=True
        )
        st.markdown("---")
        st.markdown("### 🏔️ Interactive Basin Membership")
        st.info("Adjust the 'water level' (Basin Threshold) to see how basins merge and which samples belong to each one.")
        basin_threshold = st.slider("Basin Threshold (Water Level)", 0.0, 1.0, 0.3, key="basin_threshold_interactive")
        col1, col2 = st.columns(2)
        with col1:
            basins = analyzer.detect_basins(st.session_state.landscape, basin_threshold)
            if basins is not None and basins.max() > 0:
                fig_basin = px.imshow(basins, origin="lower", title="Detected Basins Map", color_continuous_scale="Viridis")
                fig_basin.update_layout(coloraxis_showscale=False, template='plotly_dark')
                st.plotly_chart(fig_basin, use_container_width=True)
            else:
                 st.warning("No basins detected at this threshold.")
        with col2:
            if 'cluster_labels' not in st.session_state:
                st.warning("Please run a clustering analysis to see basin membership and composition.")
            elif basins is not None and basins.max() > 0:
                points_2d = st.session_state.embedded[:, :2]
                point_basin_labels = griddata(np.vstack([st.session_state.xx.ravel(), st.session_state.yy.ravel()]).T, basins.ravel(), points_2d, method='nearest')
                unique_basins = sorted([b for b in np.unique(point_basin_labels) if b > 0])
                st.metric("Number of Populated Basins", len(unique_basins))
                for b_id in unique_basins:
                    with st.expander(f"**Basin {int(b_id)}**"):
                        basin_mask = (point_basin_labels == b_id)
                        sample_indices = np.where(basin_mask)[0]
                        cluster_labels = np.array(st.session_state.get('cluster_labels', []))
                        if cluster_labels.size > 0:
                            cluster_labels_in_basin = cluster_labels[basin_mask]
                            st.markdown(f"**Contains {len(sample_indices)} samples.**")
                            if len(cluster_labels_in_basin) > 0:
                                cluster_counts = pd.Series(cluster_labels_in_basin).value_counts()
                                st.markdown("**Cluster Composition:**"); st.dataframe(cluster_counts.rename("Count"))
                        else:
                            st.markdown(f"**Contains {len(sample_indices)} samples.**")
                        all_ids = st.session_state.get('sequence_ids', [f"Sample_{i}" for i in range(len(points_2d))])
                        all_ids_list = list(all_ids)
                        if sample_indices.size > 0:
                            st.text_area("Sample IDs in Basin", ", ".join([all_ids_list[i] for i in sample_indices]), height=100, key=f"basin_text_{b_id}")

    # --- 3D Scatter (optional) ---
    if tab_offset:
        with active_tabs[1]:
            st.markdown("### 3D Embedding Scatter Plot")
            color_labels = st.session_state.get('cluster_labels', np.zeros(len(st.session_state.embedded))).astype(str)
            fig_3d = px.scatter_3d(x=st.session_state.embedded[:,0], y=st.session_state.embedded[:,1], z=st.session_state.embedded[:,2], color=color_labels, hover_name=st.session_state.get('sequence_ids'))
            fig_3d.update_layout(height=800, title="3D Scatter Plot of Embedded Data", legend_title_text='Cluster', template='plotly_dark')
            st.plotly_chart(fig_3d, use_container_width=True)

    # --- Coherence Tab ---
    with active_tabs[1 + tab_offset]:
        st.markdown("### 🧬 Coherence: Within-Cluster Concentration")
        st.info("Coherence measures how dense and well-formed individual clusters are. Higher scores are generally better.")
        if 'cluster_labels' in st.session_state and len(np.unique(st.session_state.get('cluster_labels', []))) > 1:
            points = st.session_state.embedded[:, :2]; labels = st.session_state.cluster_labels
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", f"{silhouette_score(points, labels):.3f}")
            col2.metric("Calinski-Harabasz Score", f"{calinski_harabasz_score(points, labels):.2f}")
        else: st.warning("Run a clustering analysis with at least 2 clusters to view coherence metrics.")

    # --- Separation Tab ---
    with active_tabs[2 + tab_offset]:
        st.markdown("### ↔️ Separation: Between-Cluster Distinctness")
        st.info("Separation measures how distinct and well-separated different clusters are from each other.")
        if 'cluster_labels' in st.session_state and len(np.unique(st.session_state.get('cluster_labels', []))) > 1:
            points = st.session_state.embedded[:, :2]; labels = st.session_state.cluster_labels
            unique_labels = sorted([l for l in np.unique(labels) if l != -1])
            jsd_df = analyzer.compute_jsd_matrix(points, labels)
            st.markdown("#### Overall Separation Score")
            if not jsd_df.empty:
                off_diagonal = jsd_df.values[~np.eye(jsd_df.shape[0], dtype=bool)]
                mean_jsd = np.nanmean(off_diagonal)
                st.metric("Mean JSD (Average Divergence)", f"{mean_jsd:.4f}")
            st.markdown("#### Pairwise Separation: Barrier Height (ΔU)")
            c1, c2 = st.columns(2)
            cluster1 = c1.selectbox("Cluster 1", unique_labels, index=0, key="sep_c1")
            cluster2 = c2.selectbox("Cluster 2", unique_labels, index=min(1, len(unique_labels)-1), key="sep_c2")
            if cluster1 != cluster2:
                barrier_height, _, _ = analyzer.calculate_barrier_height(st.session_state.landscape, st.session_state.xx, st.session_state.yy, points, labels, cluster1, cluster2)
                if barrier_height is not None: st.metric(f"Barrier Height (ΔU) between {cluster1} & {cluster2}", f"{barrier_height:.4f}")
                else: st.warning("Could not calculate barrier height.")
            else: st.warning("Please select two different clusters.")
            st.markdown("#### Pairwise Divergence: JSD Matrix")
            if not jsd_df.empty: st.dataframe(jsd_df.style.background_gradient(cmap='viridis').format("{:.4f}"))
        else: st.warning("Run a clustering analysis with at least 2 clusters to view separation metrics.")

    # --- Persistence Tab (existing consensus run results) ---
    with active_tabs[3 + tab_offset]:
        st.markdown("### 🔄 Persistence: Stability Under Perturbation")
        st.info("Persistence measures how stable cluster assignments are when the analysis is repeated on perturbed data. This is a key measure of robustness.")
        if 'coassoc_matrix' in st.session_state:
            st.metric("Avg. Variation of Information", f"{st.session_state.get('avg_vi', 0):.4f}")
            st.markdown("#### Co-association Heatmap")
            st.markdown("This heatmap shows the frequency (from 0 to 1) that any two samples were clustered together across all consensus runs. Well-defined blocks along the diagonal indicate stable clusters.")
            labels_for_heatmap = st.session_state.get('sequence_ids', [f'S{i}' for i in range(len(st.session_state.data))])
            fig = go.Figure(data=go.Heatmap(z=st.session_state.coassoc_matrix, x=labels_for_heatmap, y=labels_for_heatmap, colorscale='viridis'))
            fig.update_layout(title="Sample Co-clustering Frequency", xaxis_showticklabels=len(labels_for_heatmap) < 50, yaxis_showticklabels=len(labels_for_heatmap) < 50, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.markdown("### Export Partitions & Analysis Data")
            df_export = pd.DataFrame({
                'sample_id': st.session_state.get('sequence_ids', range(len(st.session_state.cluster_labels))), 
                'cluster_id': st.session_state.cluster_labels,
                'dim_1': st.session_state.embedded[:, 0],
                'dim_2': st.session_state.embedded[:, 1],
                'instability_score': st.session_state.get('instability_scores', np.nan)
            })
            st.download_button("Download Full Analysis Data (CSV)", df_export.to_csv(index=False), "full_analysis_data.csv", "text/csv", use_container_width=True, type="primary")
        else: st.warning("Run 'Consensus' clustering from the sidebar to perform a persistence analysis.")

    # --- Correlations Tab ---
    with active_tabs[4 + tab_offset]:
        st.markdown("### 📈 Correlation Explorer")
        data = st.session_state.data
        is_square = (data.ndim == 2 and data.shape[0] == data.shape[1])

        src = st.radio("Matrix source",
                       ["Features (CSV numeric columns)", "Cluster-wise (per cluster)", "Mantel: Dist vs 1−Coassoc"],
                       horizontal=True)

        if src == "Features (CSV numeric columns)":
            if is_square:
                st.warning("A square matrix is loaded (likely a distance matrix). Load feature data to compute feature correlations.")
            else:
                X = _to_numeric_df(data)
                method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
                shrink = st.toggle("Use Ledoit–Wolf shrinkage (robust for p≈n)", value=False)
                cluster_rows = st.toggle("Cluster rows/cols", value=True)
                mask_upper = st.toggle("Mask upper triangle", value=True)
                corr_df = compute_correlation_df(X, method=method, shrinkage=shrink)
                if cluster_rows:
                    corr_df = cluster_reorder(corr_df)
                fig = plot_corr_heatmap(corr_df, title=f"{method.title()} correlation", mask_upper=mask_upper)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download correlation matrix (CSV)",
                                   corr_df.to_csv(), "feature_correlations.csv", "text/csv", use_container_width=True)

        elif src == "Cluster-wise (per cluster)":
            if 'cluster_labels' not in st.session_state or len(np.unique(st.session_state.cluster_labels)) < 2:
                st.warning("Run a clustering analysis with ≥2 clusters first.")
            elif is_square:
                st.warning("Cluster-wise feature correlations require feature data (not a distance matrix).")
            else:
                X = _to_numeric_df(st.session_state.data)
                method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
                clusters = sorted([c for c in np.unique(st.session_state.cluster_labels) if c != -1])
                for c in clusters:
                    idx = np.where(st.session_state.cluster_labels == c)[0]
                    if len(idx) < 4:
                        st.info(f"Cluster {c}: too few samples ({len(idx)}) for a stable correlation estimate.")
                        continue
                    corr_df = compute_correlation_df(X.iloc[idx, :], method=method)
                    corr_df = cluster_reorder(corr_df)
                    st.markdown(f"#### Cluster {c} ({len(idx)} samples)")
                    st.plotly_chart(plot_corr_heatmap(corr_df, title=f"Cluster {c} — {method.title()}", mask_upper=True), use_container_width=True)

        else:  # Mantel
            if not is_square:
                st.warning("Mantel requires a square distance matrix input. Load a distance matrix CSV or compute one upstream.")
            else:
                D = data.astype(float)
                if 'coassoc_matrix' in st.session_state:
                    C = 1.0 - st.session_state.coassoc_matrix
                    method = st.selectbox("Mantel correlation", ["pearson", "spearman"])
                    perms = st.slider("Permutations", 199, 9999, 999, step=100)
                    with st.spinner("Running Mantel permutations…"):
                        r, p = mantel_test(D, C, n_perm=perms, method=method)
                    st.metric("Mantel r", f"{r:.3f}")
                    st.metric("p-value", f"{p:.4f}")
                else:
                    st.info("Run Consensus clustering to get the co-association matrix, then come back for Mantel.")

    # --- Cluster Distributions Tab ---
    with active_tabs[5 + tab_offset]:
        st.markdown("### 📊 Cluster Distribution Fits")
        if 'cluster_labels' not in st.session_state or len(np.unique(st.session_state.cluster_labels)) < 2:
            st.warning("Run a clustering analysis with ≥2 clusters first.")
        else:
            labels = st.session_state.cluster_labels
            pts2d = st.session_state.embedded[:, :2]

            var_choice = st.selectbox("Variable to model", ["Radial distance (embedding 2D)", "Neighbor counts within radius", "Feature column (if available)"])

            if var_choice == "Feature column (if available)":
                feat_names = st.session_state.get('feature_names', None)
                if feat_names is None:
                    st.warning("Feature names are not available (likely because a distance matrix or FASTA was loaded). Choose another variable.")
                    st.stop()
                col_name = st.selectbox("Choose feature", feat_names)
                col_idx = feat_names.index(col_name)
                data_vec = st.session_state.data[:, col_idx]
                var_type = st.selectbox("Assumed type for this feature", ["continuous", "bounded01", "count"], index=0)
                per_cluster_data = {}
                for c in sorted([x for x in np.unique(labels) if x != -1]):
                    per_cluster_data[c] = data_vec[labels == c]

            elif var_choice == "Neighbor counts within radius":
                # radius as fraction of global scale
                from sklearn.neighbors import NearestNeighbors
                all_d = scipy.spatial.distance.pdist(pts2d)
                default_r = float(np.percentile(all_d, 10))  # modest local neighborhood
                r = st.slider("Neighborhood radius (embedding units)", float(default_r*0.25), float(default_r*4.0), float(default_r), step=float(default_r*0.05))
                nn = NearestNeighbors(radius=r).fit(pts2d)
                nbrs = nn.radius_neighbors(pts2d, return_distance=False)
                counts = np.array([len(idx)-1 for idx in nbrs])  # exclude self
                per_cluster_data = {}
                for c in sorted([x for x in np.unique(labels) if x != -1]):
                    per_cluster_data[c] = counts[labels == c]
                var_type = "count"

            else:  # Radial distance
                # distance from cluster centroid in 2D
                per_cluster_data = {}
                for c in sorted([x for x in np.unique(labels) if x != -1]):
                    pts = pts2d[labels == c]
                    cen = pts.mean(axis=0)
                    d = np.sqrt(((pts - cen)**2).sum(axis=1))
                    per_cluster_data[c] = d
                var_type = "continuous"  # we will also try bounded01 view via rescale
            
            # Summary table across clusters
            rows = []
            for c, vec in per_cluster_data.items():
                fits = []
                if var_type == "continuous":
                    fits += best_fit_distributions_1d(vec, var_type="continuous")
                    # also try Beta on rescaled 0-1
                    fits += best_fit_distributions_1d(vec, var_type="bounded01")
                elif var_type == "count":
                    fits += best_fit_distributions_1d(vec, var_type="count")
                else: # bounded01
                    fits += best_fit_distributions_1d(vec, var_type="bounded01")
                best = fits[0]["name"] if fits else "N/A"
                rows.append({"Cluster": c, "n": int(len(vec)), "Best (AIC)": best, "Top-3": ", ".join([f["name"] for f in fits[:3]]) if fits else "N/A"})
            if rows:
                st.dataframe(pd.DataFrame(rows))

            # Detail view
            csel = st.selectbox("Inspect cluster", sorted([x for x in np.unique(labels) if x != -1]))
            xvec = per_cluster_data.get(csel, np.array([]))
            if len(xvec) < 5:
                st.info("Not enough data to fit distributions for this cluster.")
            else:
                fits = []
                if var_type == "continuous":
                    fits += best_fit_distributions_1d(xvec, var_type="continuous")
                    fits += best_fit_distributions_1d(xvec, var_type="bounded01")
                elif var_type == "count":
                    fits += best_fit_distributions_1d(xvec, var_type="count")
                else:
                    fits += best_fit_distributions_1d(xvec, var_type="bounded01")
                st.plotly_chart(plot_fit_overlay(xvec, fits, var_type=("count" if var_type=="count" else "continuous"), title=f"Cluster {csel} — Distribution fits"), use_container_width=True)
                if fits:
                    st.markdown("**Top fits (by AIC):**")
                    st.table(pd.DataFrame([{"Rank": i+1, "Name": f["name"], "AIC": f["aic"], "BIC": f["bic"]} for i, f in enumerate(fits[:5])]))
    
    # --- TDA (optional) ---
    tda_tab_index = 6 + tab_offset
    if OPTIONAL_LIBS['kmapper'] and len(active_tabs) > tda_tab_index:
        with active_tabs[tda_tab_index]:
            st.info("TDA Mapper creates a simplified graph of your data's shape. Use the sidebar to tune parameters.")
            if 'instability_scores' not in st.session_state:
                st.warning("Please run a 'Consensus' clustering analysis first to generate instability scores for TDA node sizing.")
            else:
                with st.spinner("Building TDA graph..."):
                    mapper = km.KeplerMapper(verbose=0)
                    data_scaled = StandardScaler().fit_transform(st.session_state.data)
                    if st.session_state.tda_lens_type == 'L2-norm':
                        lens = mapper.fit_transform(data_scaled, projection=None)
                    else:
                        pca_idx = int(st.session_state.tda_lens_type[-1]) - 1
                        lens = PCA(n_components=2, random_state=42).fit_transform(data_scaled)[:, pca_idx:pca_idx+1]
                    graph = mapper.map(
                        lens, data_scaled,
                        clusterer=DBSCAN(eps=st.session_state.tda_dbscan_eps, min_samples=3),
                        cover=km.Cover(n_cubes=st.session_state.tda_intervals, perc_overlap=st.session_state.tda_overlap / 100)
                    )
                    labels = st.session_state.get('cluster_labels', np.zeros(len(st.session_state.data)))
                    sequence_ids = st.session_state.get('sequence_ids', [f"Sample_{i}" for i in range(len(st.session_state.data))])
                    instability = st.session_state.get('instability_scores', np.zeros(len(st.session_state.data)))
                    col1, col2 = st.columns([1,3])
                    with col1:
                        node_size_metric = st.radio("Node Size Metric:", ["Number of Samples", "Mean Instability"])
                        color_map = st.selectbox("Color Map:", ["Viridis", "Plasma", "Cividis", "Rainbow", "Blues"])
                    with col2:
                        tda_fig = create_tda_graph(graph, labels, sequence_ids, instability, node_size_metric, color_map)
                        st.plotly_chart(tda_fig, use_container_width=True)
def main():
    if 'analyzer' not in st.session_state: st.session_state.analyzer = InfoscapeAnalyzer()
    analyzer = st.session_state.analyzer
    st.title("🗻 Infoscapes Studio")
    st.markdown("**An interactive tool for exploring high-dimensional data landscapes.**")
    
    run_interactive, run_persistence = render_sidebar()

    if run_interactive:
        clear_analysis_state()
        with st.spinner("Running interactive analysis..."):
            try:
                n_components = st.session_state.get('n_components', 2)
                embedded = analyzer.compute_embedding(st.session_state.data, st.session_state.embedding_method, n_components, **st.session_state.embedding_kwargs)
                if embedded is not None:
                    st.session_state.embedded = embedded
                    points_2d = embedded[:, :2]
                    x_min, x_max = points_2d[:,0].min(), points_2d[:,0].max()
                    y_min, y_max = points_2d[:,1].min(), points_2d[:,1].max()
                    ext = st.session_state.landscape_extent
                    xx, yy = np.mgrid[x_min-ext*(x_max-x_min):x_max+ext*(x_max-x_min):150j, y_min-ext*(y_max-y_min):y_max+ext*(y_max-y_min):150j]
                    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
                    if st.session_state.landscape_maker == "KDE":
                        kernel = gaussian_kde(points_2d.T, bw_method=st.session_state.kde_bandwidth)
                        density = np.reshape(kernel(grid_points.T).T, xx.shape)
                    elif st.session_state.landscape_maker == "k-NN Density":
                        density = analyzer.compute_knn_density(points_2d, grid_points, k=st.session_state.knn_k)
                        density = np.reshape(density, xx.shape)
                    elif st.session_state.landscape_maker == "GMM":
                        from sklearn.mixture import GaussianMixture
                        gmm = GaussianMixture(n_components=st.session_state.gmm_n_components, random_state=42).fit(points_2d)
                        log_density = gmm.score_samples(grid_points)
                        density = np.exp(log_density).reshape(xx.shape)
                    if st.session_state.transform_type in ["Information Height", "Potential Energy"]: landscape = -np.log(density + 1e-9)
                    elif st.session_state.transform_type == "Probability Density": landscape = density
                    elif st.session_state.transform_type == "Gradient (Cliffs)": grad_x, grad_y = np.gradient(density); landscape = np.sqrt(grad_x**2 + grad_y**2)
                    elif st.session_state.transform_type == "Laplacian of Density": landscape = -laplace(density)
                    st.session_state.update(xx=xx, yy=yy, landscape=landscape)
                    cluster_method = st.session_state.cluster_method
                    if cluster_method != "None":
                        cluster_points = points_2d
                        if cluster_method == "Consensus":
                            labels, coassoc, all_labels = analyzer.compute_consensus_clustering(cluster_points, **st.session_state.cluster_kwargs)
                            instability = analyzer.identify_unstable_samples(coassoc, labels)
                            st.session_state.update(cluster_labels=labels, coassoc_matrix=coassoc, avg_vi=analyzer.compute_variation_information(all_labels), instability_scores=instability)
                        else:
                            if cluster_method == "DBSCAN": labels = DBSCAN(eps=np.percentile(pdist(cluster_points),15)).fit_predict(cluster_points)
                            else: labels = KMeans(n_clusters=st.session_state.cluster_kwargs.get('n_clusters',4), n_init='auto').fit_predict(cluster_points)
                            st.session_state.cluster_labels = labels
            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")
                if 'embedded' in st.session_state: del st.session_state['embedded']
                return
        st.rerun()

    # <<< NEW FEATURE: AUTOMATED PERSISTENCE RUN LOGIC >>>
    if run_persistence:
        clear_analysis_state()
        partitions = {}
        progress_bar = st.sidebar.progress(0)
        num_methods = len(st.session_state.persistence_methods)
        for i, method in enumerate(st.session_state.persistence_methods):
            st.sidebar.text(f"Running {method}...")
            try:
                embedding = analyzer.compute_embedding(st.session_state.data, method, n_components=2)
                if embedding is not None:
                    labels = KMeans(n_clusters=st.session_state.persistence_k, random_state=42, n_init='auto').fit_predict(embedding)
                    partitions[method] = labels
            except Exception as e:
                st.sidebar.warning(f"Failed to run {method}: {e}")
            progress_bar.progress((i + 1) / num_methods)
        
        partitions_df = pd.DataFrame(partitions, index=st.session_state.get('sequence_ids', range(len(st.session_state.data))))
        partitions_array = partitions_df.values.T
        
        st.session_state.persistence_results = {
            'cocluster_matrix': analyzer.compute_coclustering_matrix(partitions_array),
            'mean_vi': analyzer.compute_partition_agreement(partitions_array),
            'partitions_df': partitions_df,
            'sample_ids': partitions_df.index.tolist()
        }
        st.rerun()

    render_main_content(analyzer)

if __name__ == "__main__":
    main()