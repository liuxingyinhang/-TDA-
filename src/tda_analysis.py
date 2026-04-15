import kmapper as km
from sklearn.cluster import DBSCAN
import os
import numpy as np

try:
    import umap

    LENS_TYPE = 'UMAP'
except ImportError:
    from sklearn.decomposition import PCA

    LENS_TYPE = 'PCA'


def run_mapper(data, labels, output_dir):
    print(f"\n🔮 [Step 4] 运行 TDA Mapper ({LENS_TYPE})...")

    if LENS_TYPE == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        lens = reducer.fit_transform(data)
    else:
        pca = PCA(n_components=2)
        lens = pca.fit_transform(data)

    mapper = km.KeplerMapper(verbose=0)

    graph = mapper.map(
        lens,
        data,
        clusterer=DBSCAN(eps=0.5, min_samples=2, metric='cosine'),
        cover=km.Cover(n_cubes=15, perc_overlap=0.5)
    )

    n_nodes = len(graph['nodes'])
    print(f"   ---> 📊 节点数: {n_nodes}")

    if n_nodes == 0:
        print("   ⚠️ 警告: 图谱为空，请调整 DBSCAN 参数")
        return graph

    label_map = {0: "Normal", 1: "SLE", 2: "SLE+LN"}
    try:
        tooltips = np.array([label_map.get(int(l), str(l)) for l in labels])
        html_path = os.path.join(output_dir, "sle_tda_mapper.html")

        mapper.visualize(
            graph,
            path_html=html_path,
            title=f"SLE TDA Analysis ({LENS_TYPE})",
            custom_tooltips=tooltips,
            color_values=np.array(labels),
            color_function_name="Group",
            node_color_function=np.array(['mean', 'max'])
        )
        if os.path.exists(html_path):
            inject_translation(html_path)
            print(f"   ✅ TDA 图谱已生成: {html_path}")

    except Exception as e:
        print(f"   ❌ TDA 生成失败: {e}")

    return graph


def inject_translation(html_file_path):
    js_code = """
    <script>
    const i18n={'zh':{'Cluster Details':'聚类(簇)详情','Mapper Summary':'拓扑摘要','Help':'帮助','Node Color Function':'节点着色依据','Cluster Statistics':'簇内统计信息','Projection Statistics':'投影统计信息','Above Average':'高于平均值的特征','Below Average':'低于平均值的特征','Size':'样本数量','Members':'簇内成员列表','Actions':'操作','Center viewport on node':'将视图中心对准此节点','Distribution':'分布','Nodes':'节点数','Edges':'连边数','Total Samples':'总样本数','Unique Samples':'唯一样本数','Color function':'着色依据','Node Color Functions':'节点颜色计算方式','Node Distribution':'节点分布直方图','mean':'均值','std':'标准差','median':'中位数','max':'最大值','Save config':'保存配置','Load config file':'加载配置文件','Load':'加载'},'en':{}};
    let currentLang='zh';
    function updateText(selector,textKey,context=document){const els=context.querySelectorAll(selector);els.forEach(el=>{for(let i=0;i<el.childNodes.length;i++){const node=el.childNodes[i];if(node.nodeType===3){const cleanText=node.nodeValue.trim();if(cleanText&&i18n[currentLang][cleanText]){node.nodeValue=" "+i18n[currentLang][cleanText]+" "}}}})}
    function translateStaticUI(){updateText('#tooltip_control h3','Cluster Details');updateText('#meta_control h3','Mapper Summary');updateText('#help_control h3','Help');updateText('#node_color_function_control h3','Node Color Function');updateText('#meta_content h3','Nodes');updateText('#meta_content h3','Edges');updateText('#meta_content h3','Total Samples');updateText('#meta_content h3','Unique Samples');updateText('#meta_content h3','Color function');updateText('#meta_content h3','Node Color Functions');updateText('#meta_content h3','Node Distribution');const saveBtn=document.getElementById('download-config');if(saveBtn&&currentLang=='zh')saveBtn.innerText=i18n['zh']['Save config'];const loadLabel=document.querySelector('label[for="config-file-loader"]');if(loadLabel&&currentLang=='zh')loadLabel.innerText=i18n['zh']['Load config file']}
    const observer=new MutationObserver((mutations)=>{if(currentLang==='zh'){updateText('#tooltip_content h3','Actions');updateText('#tooltip_content h3','Distribution');updateText('#tooltip_content h3','Projection Statistics');updateText('#tooltip_content h3','Cluster Statistics');updateText('#tooltip_content h3','Above Average');updateText('#tooltip_content h3','Below Average');updateText('#tooltip_content h3','Size');updateText('#tooltip_content h3','Members');updateText('#tooltip_content h2','Cluster Details');const centerBtn=document.querySelector('.center-on-node');if(centerBtn)centerBtn.innerText=i18n['zh']['Center viewport on node'];const ths=document.querySelectorAll('th small');ths.forEach(th=>{if(i18n['zh'][th.innerText])th.innerText=i18n['zh'][th.innerText]})}});
    observer.observe(document.getElementById('tooltip_content'),{childList:true,subtree:true});
    window.addEventListener('load',function(){const header=document.querySelector('.wrap-header div:last-child');if(header){const btn=document.createElement('button');btn.innerHTML='🌐 中/En';btn.className='btn';btn.style.marginLeft='10px';btn.style.backgroundColor='#2ecc71';btn.onclick=function(){if(currentLang==='zh'){currentLang='en';location.reload()}else{currentLang='zh';translateStaticUI()}};header.appendChild(btn)}setTimeout(()=>{translateStaticUI()},500)});
    </script>
    """
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if '</body>' in content:
            new_content = content.replace('</body>', js_code + '\n</body>')
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
    except Exception:
        pass