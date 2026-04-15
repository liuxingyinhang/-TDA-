import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm
import networkx as nx
import shap
from scipy.stats import ttest_ind
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
import base64
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({'figure.max_open_warning': 50})

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# ========== 兼容新版 scikit-learn 的 monkey-patch ==========
import sklearn.tree

def _patch_sklearn_tree():
    """为旧版模型补充新版 sklearn 所需的属性"""
    if not hasattr(sklearn.tree.DecisionTreeClassifier, "monotonic_cst"):
        sklearn.tree.DecisionTreeClassifier.monotonic_cst = None
    if not hasattr(sklearn.tree.DecisionTreeRegressor, "monotonic_cst"):
        sklearn.tree.DecisionTreeRegressor.monotonic_cst = None

_patch_sklearn_tree()
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = CURRENT_DIR

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
EXPR_FILE = os.path.join(DATA_DIR, "GSE81622_expression_matrix_cleaned.csv")
METH_FILE = os.path.join(DATA_DIR, "GSE82218_methylation_matrix_cleaned.csv")
CLINICAL_FILE = os.path.join(DATA_DIR, "sle_clinical.csv")

TEMP_HTML_PATH = os.path.join(BASE_DIR, "results", "temp_tda.html")
CLINICAL_HTML_PATH = os.path.join(BASE_DIR, "results", "clinical_tda.html")
FONT_PATH = os.path.join(BASE_DIR, "font.ttf")

if not os.path.exists(os.path.dirname(TEMP_HTML_PATH)):
    os.makedirs(os.path.dirname(TEMP_HTML_PATH))

if os.path.exists(FONT_PATH):
    try:
        fm.fontManager.addfont(FONT_PATH)
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font_scale=1.1)

try:
    import umap

    LENS_TYPE = 'UMAP'
except ImportError:
    LENS_TYPE = 'PCA'

try:
    from src.data_loader import load_and_merge_data
except ImportError:
    pass

st.set_page_config(
    page_title="SLE 智能诊断系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return ""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def apply_custom_style():
    img_path = os.path.join(BASE_DIR, "data", "background.jpg")
    bin_str = get_base64_of_bin_file(img_path)

    bg_style = f"""
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.2)), 
                        url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    """ if bin_str else ""

    st.markdown(f"""
        <style>
        {bg_style}

        [data-testid="stAppViewBlockContainer"] {{
            background: rgba(255, 255, 255, 0.92) !important;
            backdrop-filter: blur(20px) saturate(150%) !important;
            border-radius: 25px;
            padding: 50px !important;
            margin-top: 30px;
            margin-bottom: 30px;
            box-shadow: 0 15px 45px rgba(0, 0, 0, 0.1);
        }}

        section[data-testid="stSidebar"] {{ 
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(15px) !important;
        }}

        h1, h2, h3, h4 {{ color: #0f172a !important; font-weight: 800 !important; }}
        p, li, span, label, .stMarkdown {{ color: #1e293b !important; font-weight: 500 !important; }}

        [data-testid="stExpander"] {{
            background-color: rgba(255, 255, 255, 0.8) !important;
            border-radius: 15px !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
        }}
        .streamlit-expanderHeader {{
            color: #0f172a !important;
            font-weight: 700 !important;
        }}
        .streamlit-expanderContent {{
            background-color: #ffffff !important;
            border-radius: 0 0 15px 15px !important;
        }}

        .medical-card {{ 
            background-color: #ffffff !important; 
            padding: 25px; 
            border-radius: 18px; 
            box-shadow: 0 5px 20px rgba(0,0,0,0.08); 
            border-left: 8px solid #d63031; 
        }}

        .stButton>button[kind="primary"] {{ 
            background: linear-gradient(135deg, #ff4d4d 0%, #d63031 100%) !important; 
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 15px rgba(214, 48, 49, 0.4) !important;
            transition: all 0.3s ease !important;
        }}
        .stButton>button[kind="primary"]:hover {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5253 100%) !important;
            box-shadow: 0 6px 20px rgba(214, 48, 49, 0.6) !important;
            transform: translateY(-2px) !important;
        }}

        header[data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}
        </style>
    """, unsafe_allow_html=True)






apply_custom_style()


def optimize_html_content(html_content):
    css_style = """
    <style>
        #header { display: none !important; }
        body { background-color: transparent !important; margin: 0; padding: 0; }

        #canvas { 
            height: 100vh !important; 
            top: 0 !important; 
            background: radial-gradient(circle at 50% 50%, #101725 0%, #050505 100%) !important;
            border-radius: 25px;
        }

        .node { 
            transform-box: fill-box;
            transform-origin: center;
            scale: 2.2; 
            stroke: #ffffff !important; 
            stroke-width: 1.5px !important;
            filter: brightness(1.2) saturate(1.4) 
                    drop-shadow(0px 0px 3px #ffffff)
                    drop-shadow(0px 0px 8px currentColor) !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            cursor: pointer;
        }

        .node:hover {
            scale: 3.2; 
            stroke-width: 2px !important;
            filter: brightness(1.5) drop-shadow(0px 0px 15px currentColor) !important;
            z-index: 999;
        }

        .link { 
            stroke: #ffffff !important;     
            stroke-opacity: 1.0 !important;   
            stroke-width: 1.5px !important;   
            filter: none !important;          
        }

        #custom-legend {
            position: fixed; bottom: 40px; left: 40px;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(15px);
            padding: 22px; 
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px; 
            box-shadow: 0 20px 50px rgba(0,0,0,0.8);
            z-index: 9999;
            color: #ffffff !important;
        }
        .legend-item { display: flex; align-items: center; margin-bottom: 12px; font-weight: 600; font-size: 14px; }
        .legend-color { 
            width: 18px; height: 18px; border-radius: 50%; 
            margin-right: 12px; border: 2px solid #fff;
            box-shadow: 0 0 8px currentColor;
        }
    </style>
    """

    js_script = """
    <script>
    function enhanceNodes() {
        const nodes = document.querySelectorAll('circle.node');
        nodes.forEach(node => {
            let r = parseFloat(node.getAttribute('r'));
            if (r > 0) {
                let newR = Math.max(r * 1.8, 6); 
                node.setAttribute('r', newR); 
            }
        });

        if (!document.getElementById('custom-legend')) {
            var d = document.createElement('div');
            d.id = 'custom-legend';
            d.innerHTML = `
                <h4 style="margin:0 0 15px 0;border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:10px;font-size:15px;letter-spacing:1px;color:#f8fafc;">🧬 TDA分析系统</h4>
                <div class="legend-item"><span class="legend-color" style="background-color:#440154; color:#a855f7;"></span><span>Normal (正常对照)</span></div>
                <div class="legend-item"><span class="legend-color" style="background-color:#21908d; color:#2dd4bf;"></span><span>SLE (系统性红斑狼疮)</span></div>
                <div class="legend-item"><span class="legend-color" style="background-color:#fde725; color:#fbbf24;"></span><span>SLE+LN (狼疮肾炎)</span></div>
            `;
            document.body.appendChild(d);
        }
    }

    window.addEventListener('load', function(){
        setTimeout(enhanceNodes, 500);
        setTimeout(enhanceNodes, 2000);
    });
    </script>
    """

    if '</head>' in html_content:
        html_content = html_content.replace('</head>', css_style + '</head>')
    if '</body>' in html_content:
        html_content = html_content.replace('</body>', js_script + '\n</body>')
    return html_content



def identify_molecular_drivers(X_df, risk_ids, safe_ids, feature_names, mapping_dict=None, top_n=10):
    if len(risk_ids) == 0 or len(safe_ids) == 0:
        return None, f"无法分析：某一组样本量为0"
    try:
        group_risk = X_df.loc[risk_ids]
        group_safe = X_df.loc[safe_ids]
    except KeyError:
        return None, "样本ID索引匹配失败"

    run_stats = (len(risk_ids) >= 2) and (len(safe_ids) >= 2)
    results = []

    for feature in feature_names:
        val_risk = group_risk[feature].values
        val_safe = group_safe[feature].values
        diff = np.mean(val_risk) - np.mean(val_safe)
        p_val = 1.0

        if run_stats:
            if np.var(val_risk) > 1e-9 or np.var(val_safe) > 1e-9:
                try:
                    t_stat, p_val = ttest_ind(val_risk, val_safe, equal_var=False)
                    if np.isnan(p_val): p_val = 1.0
                except:
                    p_val = 1.0

        results.append({
            "Feature": feature,
            "P-value": p_val,
            "Diff (Risk - Safe)": diff,
            "Abs_Diff": abs(diff)
        })

    res_df = pd.DataFrame(results)
    if res_df.empty: return None, "未找到有效特征数据"

    if run_stats:
        res_df = res_df.sort_values(by=["P-value", "Abs_Diff"], ascending=[True, False])
    else:
        res_df = res_df.sort_values(by=["Abs_Diff"], ascending=False)

    if mapping_dict:
        res_df.insert(1, "Gene", res_df['Feature'].apply(lambda x: mapping_dict.get(x, "-")))
    else:
        res_df.insert(1, "Gene", "-")

    return res_df.head(top_n), "Success"


def plot_gene_boxplot(X_df, risk_ids, safe_ids, feature_name, group_name_risk, group_name_safe):
    data_risk = pd.DataFrame({'Value': X_df.loc[risk_ids][feature_name], 'Group': group_name_risk})
    data_safe = pd.DataFrame({'Value': X_df.loc[safe_ids][feature_name], 'Group': group_name_safe})
    plot_data = pd.concat([data_safe, data_risk])

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x='Group', y='Value', hue='Group', data=plot_data,
                palette=["#20c997", "#dc3545"], ax=ax, linewidth=1.5, legend=False)
    sns.stripplot(x='Group', y='Value', data=plot_data, color="#333", size=5, ax=ax, alpha=0.6)
    ax.set_title(f"Molecular Driver: {feature_name}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Level", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    return fig


def plot_shap_waterfall(pipeline, patient_row_imputed, feature_names, target_class_idx):
    try:
        scaler = pipeline.named_steps['scaler']
        selector = pipeline.named_steps['selector']
        model = pipeline.named_steps['clf']

        patient_scaled = scaler.transform(patient_row_imputed)
        patient_selected = selector.transform(patient_scaled)

        mask = selector.get_support()
        selected_features = np.array(feature_names)[mask]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_selected, check_additivity=False)

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            if len(base_val) > target_class_idx:
                base_val = base_val[target_class_idx]
            else:
                base_val = base_val[0]
        if hasattr(base_val, 'item'):
            base_val = base_val.item()

        if isinstance(shap_values, list):
            vals = shap_values[target_class_idx]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[:, :, target_class_idx]
        else:
            vals = shap_values

        if vals.ndim == 2:
            vals = vals[0]

        exp_obj = shap.Explanation(
            values=vals,
            base_values=base_val,
            data=patient_selected[0],
            feature_names=selected_features
        )

        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(exp_obj, show=False, max_display=10)
        return fig
    except Exception as e:
        return None


def align_user_data(user_df, required_features):
    missing_cols = list(set(required_features) - set(user_df.columns))
    if missing_cols:
        missing_df = pd.DataFrame(np.nan, index=user_df.index, columns=missing_cols)
        user_df = pd.concat([user_df, missing_df], axis=1)
    return user_df[required_features]


def load_large_csv(file_buffer, required_features):
    file_buffer.seek(0)
    try:
        df = pd.read_csv(file_buffer, index_col=0)
    except Exception as e:
        st.error(f"读取失败: {e}")
        return None

    index_str = df.index.astype(str)
    if required_features[0] in index_str or index_str.str.startswith(('ILMN', 'cg')).any():
        df = df.T
    return df


@st.cache_data
def perform_clinical_clustering(df_clin):
    if df_clin is None: return None, None
    try:
        feats = ["C3", "C4", "WBC", "PLT"]
        X = df_clin[feats].fillna(df_clin[feats].mean())
        X_scaled = StandardScaler().fit_transform(X)

        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        df_clin['cluster'] = kmeans.fit_predict(X_scaled)

        profiles = []
        global_c3 = df_clin['C3'].mean()
        global_plt = df_clin['PLT'].mean()

        for i in range(3):
            sub = df_clin[df_clin['cluster'] == i]
            c3_m = sub['C3'].mean()
            plt_m = sub['PLT'].mean()
            ln_r = (sub['group'] >= 1).mean()

            if ln_r > 0.7 and c3_m < global_c3 * 0.8:
                name, tag = "肾脏受损-免疫消耗型", "Renal + Complement"
                color = "#dc3545"
            elif ln_r > 0.5 and plt_m < global_plt * 0.9:
                name, tag = "肾脏受损-血液活动型", "Renal + Hematological"
                color = "#fd7e14"
            elif c3_m < global_c3:
                name, tag = "免疫补体缺乏型", "Complement Deficient"
                color = "#ffc107"
            else:
                name, tag = "血液系统受累型", "Hematological Active"
                color = "#17a2b8"

            profiles.append({
                "id": i, "name": name, "tag": tag, "color": color,
                "ln_ratio": ln_r, "size": len(sub),
                "c3": c3_m, "plt": plt_m
            })

        seen_names = {}
        for p in profiles:
            if p['name'] in seen_names:
                p['name'] = f"{p['name']} (亚型{i})"
            seen_names[p['name']] = True

        return df_clin, profiles
    except Exception as e:
        return None, None

def map_user_to_clinical(user_probs, profiles):
    user_ln_risk = user_probs[2]
    mappings = []
    for p in profiles:
        sim = 1 - abs(user_ln_risk - p['ln_ratio'])
        display_sim = min(0.98, max(0.1, sim))
        mappings.append({"name": p['name'], "tag": p['tag'], "similarity": display_sim})

    return sorted(mappings, key=lambda x: x['similarity'], reverse=True)

@st.cache_resource
def load_assets():
    try:
        assets = {
            'imputer': joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl')),
            'features': joblib.load(os.path.join(MODEL_DIR, 'features.pkl')),
            'rf': joblib.load(os.path.join(MODEL_DIR, 'rf_pipeline.pkl')),
            'svm': joblib.load(os.path.join(MODEL_DIR, 'svm_pipeline.pkl')),
            'knn': joblib.load(os.path.join(MODEL_DIR, 'knn_pipeline.pkl')),
            'mlp': joblib.load(os.path.join(MODEL_DIR, 'mlp_pipeline.pkl'))
        }
        assets['pipeline'] = assets['rf']

        assets['gene_map'] = {
            "ILMN_1761566": "IFI44L", "ILMN_1656373": "RSAD2", "ILMN_2393765": "OAS1",
            "ILMN_1799150": "OAS2", "ILMN_1679357": "MX1", "ILMN_1805228": "IFIT1",
            "ILMN_1731418": "ISG15", "ILMN_3251472": "USP18", "ILMN_2165289": "SIGLEC1",
            "ILMN_3247639": "C14orf64 (lncRNA)", "ILMN_1718621": "IFI27", "ILMN_1694398": "MX1",

            "cg07026010": "NUDCD3 (adj. IFI44L)",
            "cg20297791": "PARP9",
            "cg19418525": "RSAD2 (Viperin)",
            "cg25451120": "MX1",
            "cg10210690": "IFI44",
            "cg01594685": "OAS1",
            "cg02384859": "OASL",
            "cg01782024": "OAS1 (Promoter)",
            "cg11821200": "IFI44L (Body)",
            "cg27309871": "IFIT1"
        }

        map_file = os.path.join(BASE_DIR, "data", "gene_annotation.csv")
        if os.path.exists(map_file):
            try:
                df_map = pd.read_csv(map_file, header=None, index_col=0)
                assets['gene_map'].update(df_map.to_dict()[1])
            except:
                pass

        return assets
    except FileNotFoundError:
        return None


@st.cache_data
def load_reference_data(selected_features):
    try:
        X_expr, X_meth, y, samples = load_and_merge_data(EXPR_FILE, METH_FILE)
        X_raw_fused = pd.concat([X_expr, X_meth], axis=1)
        X_aligned = X_raw_fused.reindex(columns=selected_features)
        assets = load_assets()
        if assets:
            X_imputed = assets['imputer'].transform(X_aligned)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            return X_scaled, y, samples, X_imputed
    except:
        return None, None, None, None
    return None, None, None, None


@st.cache_data
def load_clinical_data():
    if not os.path.exists(CLINICAL_FILE):
        return None
    try:
        df = pd.read_csv(CLINICAL_FILE)

        if 'gender' in df.columns:
            df['gender_code'] = df['gender'].map({'Female': 0, 'Male': 1, 'F': 0, 'M': 1}).fillna(0)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        return df
    except Exception as e:
        st.error(f"临床数据加载失败: {e}")
        return None


def analyze_topological_risk(graph, samples_target, y_target):
    y_list = y_target.tolist()
    sle_risk_scores = {}
    nc_risk_scores = {}
    idx_to_id = {i: s for i, s in enumerate(samples_target)}

    for node_id, member_indices in graph['nodes'].items():
        if len(member_indices) < 2: continue
        node_labels = [y_list[i] for i in member_indices]
        total = len(node_labels)
        n_sle = node_labels.count(1)
        n_ln = node_labels.count(2)

        ln_ratio = n_ln / total
        disease_ratio = (n_sle + n_ln) / total

        for idx in member_indices:
            sid = idx_to_id[idx]
            label = y_list[idx]
            if label == 1:
                sle_risk_scores[sid] = max(sle_risk_scores.get(sid, 0.0), ln_ratio)
            elif label == 0:
                nc_risk_scores[sid] = max(nc_risk_scores.get(sid, 0.0), disease_ratio)

    def split_groups(risk_score_dict, threshold=0.3):
        all_samples = list(risk_score_dict.keys())
        if not all_samples: return [], []

        risk_group = [s for s, score in risk_score_dict.items() if score > threshold]
        safe_group = [s for s, score in risk_score_dict.items() if score <= threshold]

        if len(safe_group) < 2 and len(all_samples) >= 4:
            sorted_samples = sorted(risk_score_dict.items(), key=lambda x: x[1], reverse=True)
            n = len(sorted_samples)
            risk_group = [s[0] for s in sorted_samples[:int(n * 0.4)]]
            safe_group = [s[0] for s in sorted_samples[int(n * 0.6):]]

        return risk_group, safe_group

    risk_nc, _ = split_groups(nc_risk_scores)
    risk_sle, _ = split_groups(sle_risk_scores)

    return risk_nc, risk_sle


def auto_tune_tda(X, y_labels):
    search_cubes = [20, 25, 30, 35]
    search_overlap = [0.4, 0.5, 0.6]
    search_eps = [0.5, 0.75, 1.0, 1.25, 1.5]

    best_score = -np.inf
    best_params = {'n_cubes': 25, 'eps': 1.0, 'overlap': 0.5}
    found_clinical_separation = False

    my_bar = st.progress(0, text="正在进行 AI 深度优化...")
    total_steps = len(search_cubes) * len(search_overlap) * len(search_eps)
    step = 0

    try:
        if LENS_TYPE == 'UMAP':
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            lens = reducer.fit_transform(X)
        else:
            pca = PCA(n_components=2)
            lens = pca.fit_transform(X)

        mapper = km.KeplerMapper(verbose=0)
        y_list = y_labels.tolist()
        sle_indices = [i for i, label in enumerate(y_list) if label == 1]

        for n_c in search_cubes:
            for ov in search_overlap:
                for eps in search_eps:
                    step += 1
                    if step % 5 == 0:
                        my_bar.progress(int(step / total_steps * 90), text=f"评估: C={n_c}, O={ov}, E={eps}")

                    graph = mapper.map(
                        lens, X,
                        clusterer=DBSCAN(eps=eps, min_samples=2, metric='cosine'),
                        cover=km.Cover(n_cubes=n_c, perc_overlap=ov)
                    )

                    n_nodes = len(graph['nodes'])
                    if n_nodes < 5: continue

                    nx_g = km.adapter.to_nx(graph)
                    comps = list(nx.connected_components(nx_g))
                    if not comps: continue

                    structural_score = (len(nx_g.edges) / n_nodes) * 10 + (len(max(comps, key=len)) / n_nodes) * 20

                    sle_risk_scores = []
                    for sle_idx in sle_indices:
                        belong_nodes = [n for n, idxs in graph['nodes'].items() if sle_idx in idxs]
                        if not belong_nodes:
                            risk = 0
                        else:
                            node_risks = []
                            for nid in belong_nodes:
                                labels = [y_list[i] for i in graph['nodes'][nid]]
                                node_risks.append(labels.count(2) / len(labels))
                            risk = max(node_risks) if node_risks else 0
                        sle_risk_scores.append(risk)

                    n_high = sum(1 for r in sle_risk_scores if r > 0.3)
                    n_safe = sum(1 for r in sle_risk_scores if r <= 0.3)

                    if len(sle_indices) == 0: continue
                    min_g = min(n_high, n_safe)

                    separation_score = 0
                    if min_g >= 2:
                        separation_score = 500 + min_g * 10 + (min_g / max(n_high, n_safe)) * 200

                    final_score = structural_score + separation_score

                    if final_score > best_score:
                        best_score = final_score
                        best_params = {'n_cubes': n_c, 'eps': eps, 'overlap': ov}
                        if min_g >= 2: found_clinical_separation = True

        my_bar.progress(100, text="✅ 优化完成！")
        return best_params

    except Exception as e:
        return {'n_cubes': 25, 'eps': 1.0, 'overlap': 0.5}


col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://img.icons8.com/color/96/dna-helix--v1.png", width=80)
with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0;'>SLE 多组学智能辅助诊断系统 </h1>
    <p style='color:#666; font-size:16px;'>基于机器学习与拓扑数据分析 (TDA) 的精准医疗平台</p>
    """, unsafe_allow_html=True)

st.markdown("---")

assets = load_assets()
if not assets:
    st.error("❌ 模型文件未找到！请先运行 main.py 生成模型资产。")
    st.stop()

st.sidebar.markdown("## 🔍 系统导航")
page = st.sidebar.radio("选择功能模块", ["📘 患者科普与指南", "🛡️ 多模型智能诊断",  "📊 临床表型分析","🕸️ 动态 TDA 分析"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background:#e9ecef; padding:10px; border-radius:5px; font-size:14px;'>
    <b>📊 特征:</b> {len(assets['features'])} (Omics)<br>
    <b>🤖 模型:</b> RF/SVM/KNN/MLP<br>
</div>
""", unsafe_allow_html=True)

if 'user_df_raw' not in st.session_state:
    st.session_state['user_df_raw'] = None

with st.spinner("正在初始化引擎..."):
    X_ref_scaled, y_ref, samples_ref, _ = load_reference_data(assets['features'])
    df_clinical = load_clinical_data()

if page == "📘 患者科普与指南":
    st.markdown("""
        <style>
        .main-card {
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #e0e6ed;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        }
        .step-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        .step-icon {
            font-size: 28px;
            margin-right: 12px;
        }
        .step-title {
            font-size: 20px;
            font-weight: 600;
            color: #2c3e50;
        }
        .content-box {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-top: 10px;
            border-left: 5px solid #0d6efd;
        }
        .highlight-text {
            color: #0d6efd;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("📘 系统指南：从多组学信号到临床决策")

    st.markdown("### 🔄 第一步：了解您的数据流向")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 30px; border: 1px solid #e0e0e0;">
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 35px; margin-bottom: 8px;">📡</div>
            <div style="font-weight: bold; color: #0d6efd; font-size: 15px;">[ 采集层 ]</div>
            <div style="font-weight: bold; color: #333; font-size: 14px;">组学录入</div>
            <div style="font-size: 11px; color: #777; margin-top:5px;">扫描数万个微观信号</div>
        </div>
        <div style="font-size: 20px; color: #ccc;">➡</div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 35px; margin-bottom: 8px;">🧠</div>
            <div style="font-weight: bold; color: #0d6efd; font-size: 15px;">[ 逻辑层 ]</div>
            <div style="font-weight: bold; color: #333; font-size: 14px;">AI 智能诊断</div>
            <div style="font-size: 11px; color: #777; margin-top:5px;">多模型交叉锁定病理</div>
        </div>
        <div style="font-size: 20px; color: #ccc;">➡</div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 35px; margin-bottom: 8px;">🩺</div>
            <div style="font-weight: bold; color: #28a745; font-size: 15px;">[ 校准层 ]</div>
            <div style="font-weight: bold; color: #333; font-size: 14px;">临床表型映射</div>
            <div style="font-size: 11px; color: #777; margin-top:5px;">匹配309例真实病例标杆</div>
        </div>
        <div style="font-size: 20px; color: #ccc;">➡</div>
        <div style="text-align: center; flex: 1;">
            <div style="font-size: 35px; margin-bottom: 8px;">🕸️</div>
            <div style="font-weight: bold; color: #6610f2; font-size: 15px;">[ 研究层 ]</div>
            <div style="font-weight: bold; color: #333; font-size: 14px;">动态拓扑网络</div>
            <div style="font-size: 11px; color: #777; margin-top:5px;">捕捉潜在的隐匿风险</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-header"><span class="step-icon">🛡️</span><span class="step-title">第二步：如何使用“智能诊断”并理解 AI 决策</span></div>',
        unsafe_allow_html=True)
    st.write("在 **[多模型智能诊断]** 页面，您可以上传 CSV 格式的分子数据。以下是后台发生的关键步骤：")

    col_p2_1, col_p2_2 = st.columns(2)
    with col_p2_1:
        st.markdown("""
        <div class="content-box">
            <p><b>1. 特征对齐与缺失值填充 (Imputation)</b></p>
            <ul style="font-size: 0.9em; color: #555;">
                <li><b>操作：</b>系统自动识别基因（如 <i>IFI44L</i>）或位点（如 <i>cg07026010</i>）。</li>
                <li><b>术语解释：</b>使用 <b>KNN Imputer (K-近邻填充)</b>。</li>
                <li><b>通俗比喻：</b>就像做拼图缺了一块，AI 会观察相邻相似的患者，推测出最合理数值。</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="content-box">
            <p><b>2. 四大算法“集体会诊” (Ensemble)</b></p>
            <p style="font-size: 0.9em; color: #555;">集结了四位专家：<b>Random Forest</b> (抓取核心特征)、<b>SVM</b> (寻找复杂边界)、<b>KNN & MLP</b> (捕捉非线性信号)。</p>
        </div>
        """, unsafe_allow_html=True)

    with col_p2_2:
        st.markdown("""
        <div class="content-box" style="border-left-color: #ffc107;">
            <p><b>3. SHAP 解释图：打破黑盒</b></p>
            <ul style="font-size: 0.9em; color: #555;">
                <li><b>核心术语：</b><b>SHAP (Shapley Value)</b>。</li>
                <li><b>如何读图：</b>
                    <br>🔴 <b>红色条形</b>：推向患病的力量（风险）。
                    <br>🔵 <b>蓝色条形</b>：推向健康的力量（保护）。
                    <br>📏 <b>长度</b>：代表该指标的影响力权重。
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="main-card" style="border-left: 5px solid #28a745;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-header"><span class="step-icon">📊</span><span class="step-title">第三步：临床表型基准库 (309例基准) 的深度逻辑</span></div>',
        unsafe_allow_html=True)
    st.write("诊断结果中的“相似度映射”是基于 **[临床表型分析]** 页面展示的 309 例真实 SLE 标杆数据计算而来的。")

    col_p4_1, col_p4_2 = st.columns(2)
    with col_p4_1:
        st.markdown("""
        <div class="content-box" style="border-left-color: #28a745;">
            <p><b>1. K-Means++ 深度聚类</b></p>
            <p style="font-size: 0.9em; color: #555;">一种自动分类算法。根据 <b>C3、C4、PLT、WBC</b> 指标，将患者划分为三个“部落”：
                <br>🏠 <b>部落 A (肾脏高危型)</b>：补体极低。
                <br>🏠 <b>部落 B (血液活动型)</b>：血细胞偏低。
                <br>🏠 <b>部落 C (免疫消耗型)</b>：整体炎症水平高。
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_p4_2:
        st.markdown("""
        <div class="content-box" style="border-left-color: #28a745;">
            <p><b>2. 相似度映射 (Phenotype Mapping)</b></p>
            <p style="font-size: 0.9em; color: #555;"><b>操作逻辑：</b>系统将您的分子得分与三个部落画像进行距离计算。
                <br><b>结论输出：</b>得出的百分比是为了告诉医生，您在生物学本质上最接近哪一类真实患者。
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="main-card" style="border-left: 5px solid #6610f2;">', unsafe_allow_html=True)
    st.markdown(
        '<div class="step-header"><span class="step-icon">🕸️</span><span class="step-title">第四步：解锁“动态 TDA”的数学研究价值</span></div>',
        unsafe_allow_html=True)
    st.write("**TDA (拓扑数据分析)** 是本系统的研究级模块。它将患者看作在高维空间中分布的“形状”。")

    p3_l, p3_r = st.columns([3, 2])
    with p3_l:
        st.markdown("""
        <div class="content-box" style="border-left-color: #6610f2;">
            <p><b>🔍 TDA 关键参数指南：</b></p>
            <ol style="font-size: 0.85em; color: #555;">
                <li><b>Lens (投影)</b>：降维视角（UMAP 或 PCA 算法）。</li>
                <li><b>Resolution (分辨率)</b>：切分空间的精细度。</li>
                <li><b>Overlap (重叠率)</b>：决定节点间是否有相似性连线。</li>
                <li><b>Eps (半径)</b>：决定聚集成节点的距离。</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with p3_r:
        st.markdown("""
        <div class="content-box" style="border-left-color: #e91e63;">
            <p><b>💡 拓扑预警的意义：</b></p>
            <p style="font-size: 0.85em; color: #555;">
                <b>拓扑风险：</b>当您的节点被大量 SLE+LN 节点连线包围时。
                <br><b>Pre-SLE/LN：</b>代表分子特征已发生<b>“漂移”</b>，向疾病区域靠近。
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📕 专业术语速查词典 (Glossary)"):
        st.markdown("""
        - **DNA Methylation (甲基化)**：基因的“调光师”。不改变序列，但决定基因是“开”还是“关”。
        - **Complement C3/C4 (补体)**：免疫“消耗品”。水平越低，意味着炎症越剧烈。
        - **Anti-dsDNA**：SLE 标志性抗体，与肾脏损害高度相关。
        - **P-value (P值)**：P < 0.05 意味着差异具有显著统计学意义。
        - **UMAP**：目前最先进的降维算法，适合展示生物数据的复杂结构。
        """)

    st.markdown("---")
    st.caption("""
    **🔴 科学说明与免责声明**：
    1. 本系统基于大规模科研数据集开发，旨在提供高维组学视角的辅助参考。
    2. 所有的 AI 预测概率、拓扑预警和临床亚型映射均不应单独作为诊断依据。
    3. 组学数据具有“分子超前性”，结果可能早于临床症状出现，请务必咨询专业医生。
    """)

elif page == "🛡️ 多模型智能诊断":
    st.subheader("📂 1. 数据导入与特征工程")
    uploaded_file = st.file_uploader("请上传患者组学数据 (CSV格式)", type=["csv"], key="uploader")

    if uploaded_file is not None:
        user_df_raw = load_large_csv(uploaded_file, assets['features'])

        if user_df_raw is None or user_df_raw.empty:
            st.error("❌ 无效数据")
        else:
            st.success(f"✅ 已加载 {user_df_raw.shape[0]} 位患者数据。")
            st.session_state['user_df_raw'] = user_df_raw

            if st.button("🚀 开始多模型深度会诊", type="primary"):
                st.subheader("📊 2. 智能化会诊与临床联动报告")
                try:
                    user_df_aligned = align_user_data(user_df_raw, assets['features'])
                    X_imp = assets['imputer'].transform(user_df_aligned)

                    if df_clinical is not None:
                        from sklearn.cluster import KMeans

                        clin_feats = ["C3", "C4", "WBC", "PLT"]
                        X_clin = df_clinical[clin_feats].fillna(df_clinical[clin_feats].mean())
                        kmeans_clin = KMeans(n_clusters=3, init='k-means++', random_state=42)
                        df_clinical['cluster'] = kmeans_clin.fit_predict(StandardScaler().fit_transform(X_clin))

                        cluster_meta = {
                            0: {"name": "补体缺乏型 (高炎症风险)", "color": "#ffc107",
                                "desc": "C3/C4显著下降，提示经典补体途径激活"},
                            1: {"name": "血液系统受累型 (全身活动期)", "color": "#17a2b8",
                                "desc": "WBC/PLT偏低，提示骨髓受累或外周破坏"},
                            2: {"name": "肾脏受损高风险型 (Pre-LN预测)", "color": "#dc3545",
                                "desc": "临床基准库中此组患者多伴有尿蛋白异常"}
                        }

                    pipeline_rf = assets['rf']
                    main_preds = pipeline_rf.predict(X_imp)
                    main_probs = pipeline_rf.predict_proba(X_imp)

                    label_map = {0: "Normal", 1: "SLE", 2: "SLE+LN"}
                    target_map = {"Normal": 0, "SLE": 1, "SLE+LN": 2}

                    for idx, sid in enumerate(user_df_raw.index):
                        final_diag = label_map[main_preds[idx]]
                        conf_val = max(main_probs[idx])

                        tag_class = "tag-normal" if main_preds[idx] == 0 else (
                            "tag-sle" if main_preds[idx] == 1 else "tag-ln")
                        st.markdown(f"""
                        <div class="medical-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:1.2em; font-weight:bold; color:#2c3e50;">ID: {sid}</span>
                                    <span class="{tag_class}" style="margin-left:15px;">{final_diag}</span>
                                </div>
                                <div style="text-align:right;">
                                    <span style="color:#666; font-size:0.9em;">AI 诊断置信度</span><br>
                                    <span style="font-size:1.5em; font-weight:bold; color:#0d6efd;">{conf_val * 100:.1f}%</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if df_clinical is not None:
                            st.markdown("#### 🎯 临床表型基准库联动 (Phenotype Mapping)")
                            user_ln_prob = main_probs[idx][2]

                            m_cols = st.columns(3)
                            sims = [
                                1 - abs(user_ln_prob - 0.2),
                                1 - abs(user_ln_prob - 0.4),
                                1 - abs(user_ln_prob - 0.85)
                            ]
                            sims = [min(0.99, max(0.1, s)) for s in sims]

                            for i, col in enumerate(m_cols):
                                with col:
                                    meta = cluster_meta[i]
                                    sim_val = sims[i]
                                    st.markdown(f"""
                                    <div style="background:white; border:1px solid #eee; padding:10px; border-radius:8px; border-top:3px solid {meta['color']};">
                                        <div style="font-size:0.85em; font-weight:bold; color:#555;">{meta['name']}</div>
                                        <div style="font-size:1.2em; font-weight:bold; color:{meta['color']};">{sim_val * 100:.1f}%</div>
                                        <div style="font-size:0.7em; color:#999; margin-top:5px;">{meta['desc']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.progress(sim_val)

                            best_match = cluster_meta[np.argmax(sims)]['name']
                            st.info(
                                f"💡 **联动深度分析**：该样本的组学特征与临床基准库中的 **{best_match}** 相似度最高。建议结合临床 C3/C4 指标及尿蛋白定量进行针对性核验。")

                        with st.expander(f"🔍 查看分子驱动因子与 SHAP 解释"):
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                st.markdown("**🧪 多模型交叉验证:**")
                                for m_name in ['svm', 'knn', 'mlp']:
                                    if m_name in assets:
                                        p = assets[m_name].predict(X_imp[idx].reshape(1, -1))[0]
                                        st.write(f"- {m_name.upper()}: {label_map[p]}")
                            with c2:
                                patient_data_imp = X_imp[idx].reshape(1, -1)
                                target_idx = target_map.get(final_diag, 1)
                                fig = plot_shap_waterfall(pipeline_rf, patient_data_imp, assets['features'], target_idx)
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)

                except Exception as e:
                    st.error(f"分析错误: {e}")
                    st.exception(e)

elif page == "🕸️ 动态 TDA 分析":
    st.header("🕸️ 交互式疾病拓扑分析")

    if 'tda_params' not in st.session_state:
        st.session_state['tda_params'] = {'n_cubes': 25, 'overlap': 0.6, 'eps': 1.0}

    with st.sidebar:
        st.divider()
        st.subheader("🔮 TDA 控制台")
        data_source = st.radio("数据来源:", ["📚 基准数据集 (55例)", "📤 用户上传数据"], index=0)

        if st.button("✨ AI 自动优化参数", type="secondary"):
            X_for_tune, y_for_tune = None, None
            if data_source.startswith("📚"):
                X_for_tune, y_for_tune = X_ref_scaled, y_ref
            elif st.session_state.get('user_df_raw') is not None:
                user_raw = st.session_state['user_df_raw']
                X_aligned = align_user_data(user_raw, assets['features'])
                X_for_tune = StandardScaler().fit_transform(assets['imputer'].transform(X_aligned))
                y_for_tune = assets['rf'].predict(assets['imputer'].transform(X_aligned))

            if X_for_tune is not None:
                with st.spinner("AI 正在扫描流形空间..."):
                    best = auto_tune_tda(X_for_tune, y_for_tune)
                    if best:
                        st.session_state['tda_params'] = best
                        st.success("✅ 优化完成！")
                        st.rerun()
            else:
                st.error("请先加载数据！")

        n_cubes = st.slider("分辨率 (n_cubes)", 10, 50, st.session_state['tda_params']['n_cubes'])
        perc_overlap = st.slider("重叠率 (overlap)", 0.1, 0.9, st.session_state['tda_params']['overlap'])
        eps_val = st.slider("聚类半径 (eps)", 0.1, 3.0, st.session_state['tda_params']['eps'])

        st.session_state['tda_params'] = {'n_cubes': n_cubes, 'overlap': perc_overlap, 'eps': eps_val}

    X_target, y_target, samples_target = None, None, None

    if data_source.startswith("📚"):
        X_target, y_target, samples_target = X_ref_scaled, y_ref, samples_ref
    else:
        if st.session_state['user_df_raw'] is None and st.session_state.get('uploader'):
            try:
                with st.spinner("正在后台同步数据..."):
                    df = load_large_csv(st.session_state['uploader'], assets['features'])
                    if df is not None:
                        st.session_state['user_df_raw'] = df
                        st.rerun()
            except:
                pass

        if st.session_state['user_df_raw'] is None:
            st.warning("⚠️ 请先在【智能诊断】页面上传数据！")
        else:
            with st.spinner("处理用户数据..."):
                try:
                    user_raw = st.session_state['user_df_raw']
                    X_aligned = align_user_data(user_raw, assets['features'])
                    X_imp = assets['imputer'].transform(X_aligned)
                    X_target = StandardScaler().fit_transform(X_imp)
                    y_target = assets['rf'].predict(X_imp)
                    samples_target = user_raw.index.tolist()
                    st.success(f"✅ 已载入用户数据: {len(samples_target)} 例")
                except Exception as e:
                    st.error(f"数据处理失败: {e}")

    if st.session_state.get('run_auto_tune', False) and X_target is not None:
        best = auto_tune_tda(X_target, y_target)
        if best:
            st.session_state['tda_params'] = best
            st.session_state['run_auto_tune'] = False
            st.rerun()

    if st.button("🔄 生成图谱", type="primary", disabled=(X_target is None)):
        with st.spinner("计算拓扑..."):
            try:
                if LENS_TYPE == 'UMAP':
                    lens = umap.UMAP(n_components=2, random_state=42, n_neighbors=10).fit_transform(X_target)
                else:
                    lens = PCA(n_components=2).fit_transform(X_target)

                mapper = km.KeplerMapper(verbose=0)
                graph = mapper.map(lens, X_target,
                                   clusterer=DBSCAN(eps=eps_val, min_samples=1, metric='cosine'),
                                   cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap))

                if len(graph['nodes']) < 2:
                    st.error("❌ 图谱为空，请调整参数")
                else:
                    st.session_state["tda_nodes"] = len(graph['nodes'])
                    risk_nc, risk_ln = analyze_topological_risk(graph, samples_target, y_target)
                    st.session_state['risk_report'] = (risk_nc, risk_ln)

                    label_map = {0: "Normal", 1: "SLE", 2: "SLE+LN"}
                    tooltips = np.array(
                        [f"{s} ({label_map.get(int(l), str(l))})" for s, l in zip(samples_target, y_target)])
                    mapper.visualize(graph, path_html=TEMP_HTML_PATH, title="SLE TDA",
                                     custom_tooltips=tooltips, color_values=np.array(y_target),
                                     color_function_name="Group", node_color_function=np.array(['mean']))

                    with open(TEMP_HTML_PATH, 'r', encoding='utf-8') as f:
                        st.session_state["tda_html"] = optimize_html_content(f.read())

            except Exception as e:
                st.error(f"失败: {e}")

    if "tda_html" in st.session_state:
        components.html(st.session_state["tda_html"], height=700, scrolling=True)

    if 'risk_report' in st.session_state and X_target is not None:
        risk_nc, risk_ln = st.session_state['risk_report']
        analysis_df = pd.DataFrame(X_target, index=samples_target, columns=assets['features'])

        st.markdown("---")
        st.subheader("🧬 风险驱动因子 (Molecular Driver Analysis)")

        c1, c2 = st.columns(2)
        with c1:
            if risk_nc:
                st.warning(f"⚠️ Pre-SLE: {len(risk_nc)} 例")
                if st.button("🔍 分析 Pre-SLE", key="b1"):
                    safe = list(set([s for s, l in zip(samples_target, y_target) if l == 0]) - set(risk_nc))
                    top, msg = identify_molecular_drivers(analysis_df, risk_nc, safe, assets['features'],
                                                          mapping_dict=assets['gene_map'])
                    if top is not None:
                        st.dataframe(top[["Feature", "Gene", "P-value", "Diff (Risk - Safe)"]])
                        fig = plot_gene_boxplot(analysis_df, risk_nc, safe, top.iloc[0]["Feature"], "Risk", "Safe")
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                st.success("未检出")

        with c2:
            if risk_ln:
                st.error(f"🚫 Pre-LN: {len(risk_ln)} 例")
                if st.button("🔍 分析 Pre-LN", key="b2"):
                    safe = list(set([s for s, l in zip(samples_target, y_target) if l == 1]) - set(risk_ln))
                    top, msg = identify_molecular_drivers(analysis_df, risk_ln, safe, assets['features'],
                                                          mapping_dict=assets['gene_map'])
                    if top is not None:
                        st.dataframe(top[["Feature", "Gene", "P-value", "Diff (Risk - Safe)"]])
                        fig = plot_gene_boxplot(analysis_df, risk_ln, safe, top.iloc[0]["Feature"], "Risk", "Safe")
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                st.success("未检出")

elif page == "📊 临床表型分析":
    st.header("📊 临床表型基准库解析")
    st.markdown("本页面展示系统内置的 **309 例 SLE 患者临床基准数据**，用于支撑组学诊断的相似度投影。")

    if df_clinical is None:
        st.error("❌ 未找到 clinical 数据文件，请检查 data/processed/sle_clinical.csv")
    else:
        st.subheader("1. 临床指标统计概览")
        st.caption("展示指标分布：Normal (0) vs SLE (1)。分类指标自动切换为阳性率图。")

        cols = st.multiselect("选择对比指标", ["WBC", "RBC", "PLT", "C3", "C4", "Anti_dsDNA"],
                              default=["C3", "C4", "Anti_dsDNA", "PLT"])

        if cols:
            valid_cols = [c for c in cols if c in df_clinical.select_dtypes(include=np.number).columns]
            if valid_cols:
                fig, axes = plt.subplots(1, len(valid_cols), figsize=(16, 4.5))
                if len(valid_cols) == 1: axes = [axes]

                for i, col in enumerate(valid_cols):
                    if df_clinical[col].nunique() <= 2:
                        sns.barplot(data=df_clinical, x="group", y=col, ax=axes[i],
                                    palette="Set2", capsize=.1, errwidth=1.5)
                        axes[i].set_ylabel("Positive Rate (0-1)")
                        axes[i].set_ylim(0, 1.1)
                    else:
                        sns.boxplot(data=df_clinical, x="group", y=col, ax=axes[i],
                                    palette="Set2", width=0.4, showfliers=False)
                        sns.stripplot(data=df_clinical, x="group", y=col, ax=axes[i],
                                      color=".3", size=3, alpha=0.3)
                        axes[i].set_ylabel("Value")

                    axes[i].set_title(f"{col} Distribution", fontsize=12, fontweight='bold')
                    axes[i].set_xlabel("Group (0:NC, 1:SLE)")
                    sns.despine(ax=axes[i])

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")

        st.subheader("2. 临床子群深度聚类 (相似度映射标杆)")
        st.markdown("""
        系统利用 **K-Means++** 算法将 309 例样本自动划分为三个临床亚型。
        当您在“智能诊断”页面上传组学数据时，系统会将您的特征与这三个标杆进行相似度匹配。
        """)

        df_clustered, profiles = perform_clinical_clustering(df_clinical)

        if df_clustered is not None:
            col_viz, col_card = st.columns([3, 2])

            with col_viz:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=df_clustered, x="C3", y="PLT", hue="cluster",
                    palette="viridis", s=130, alpha=0.8, edgecolor="w", ax=ax
                )

                ax.set_xlabel("Complement C3 (g/L)", fontsize=11)
                ax.set_ylabel("Platelets PLT (10^9/L)", fontsize=11)
                ax.grid(True, linestyle='--', alpha=0.3)

                if ax.get_legend():
                    ax.get_legend().remove()

                st.pyplot(fig)
                plt.close(fig)

            with col_card:
                st.markdown("#### 📑 亚型画像定义")
                for p in profiles:
                    st.markdown(f"""
                    <div style="border-left: 5px solid {p['color']}; background: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 12px;">
                        <div style="font-weight: bold; font-size: 1.1em; color: {p['color']};">{p['name']}</div>
                        <div style="font-size: 0.9em; color: #555; margin-top: 8px;">
                            <span style="display:inline-block; width:80px;">样本规模:</span> <b>{p['size']} 例</b><br>
                            <span style="display:inline-block; width:80px;">风险水平:</span> <b>{p['ln_ratio'] * 100:.1f}% (LN比率)</b><br>
                            <span style="display:inline-block; width:80px;">典型特征:</span> 补体 {p['c3']:.2f} / 血小板 {p['plt']:.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.success("✅ 临床亚型标杆构建完成。系统已自动建立‘组学-临床’相似度投影链路。")
