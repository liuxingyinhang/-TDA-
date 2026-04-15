import pandas as pd
import os
import re

INPUT_FILE = r"C:\Users\liuxh\PycharmProjects\TDA\data\manual\methylation\GSE82218_series_matrix.txt"

OUTPUT_DIR = r"C:\Users\liuxh\PycharmProjects\TDA\data\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "GSE82218_methylation_matrix_cleaned.csv")


def process_gse82218(file_path):
    print(f"🚀 [GSE82218] 开始处理文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件未找到! 请检查路径是否正确。")
        return

    sample_map = {}
    start_line = 0

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        accessions = []
        titles = []

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith("!Sample_geo_accession"):
                accessions = [x.strip().strip('"') for x in line.split('\t')[1:]]

            elif line.startswith("!Sample_title"):
                titles = [x.strip().strip('"') for x in line.split('\t')[1:]]

            elif line.startswith("!series_matrix_table_begin"):
                start_line = i + 1
                break

        if not accessions or not titles:
            print("❌ 未能在文件头部找到样本映射信息，请检查文件内容是否完整。")
            return

        clean_titles = []
        print("🔍 正在提取样本分组名称...")
        for t in titles:
            match = re.search(r'\[(.*?)\]', t)
            if match:
                clean_titles.append(match.group(1))
            else:
                clean_titles.append(t)

        sample_map = dict(zip(accessions, clean_titles))
        print(f"✅ 成功建立映射 (前3个示例): {list(sample_map.items())[:3]}")

        print("⏳ 正在加载大数据矩阵 (可能需要几秒钟)...")
        df = pd.read_csv(file_path, sep='\t', skiprows=start_line, index_col=0)

        df.columns = [c.strip('"') for c in df.columns]
        if isinstance(df.index[0], str):
            df.index = [i.strip('"') for i in df.index]

        df = df.rename(columns=sample_map)

        if df.index[-1] == '!series_matrix_table_end':
            df = df.iloc[:-1]

        df = df.apply(pd.to_numeric, errors='coerce')

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        df.to_csv(OUTPUT_FILE)
        print("-" * 40)
        print(f"🎉 处理成功!")
        print(f"📄 输出文件: {OUTPUT_FILE}")
        print(f"📊 数据维度: {df.shape} (Rows=CpG, Cols=Samples)")
        print(f"👀 数据预览 (列名已清洗):")
        print(df.head())
        print("-" * 40)

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")


if __name__ == "__main__":
    process_gse82218(INPUT_FILE)