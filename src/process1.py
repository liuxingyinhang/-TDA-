import pandas as pd
import os

INPUT_FILE = r"C:\Users\liuxh\PycharmProjects\TDA\data\manual\metabolome\GSE81622_series_matrix.txt"

OUTPUT_DIR = r"C:\Users\liuxh\PycharmProjects\TDA\data\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "GSE81622_expression_matrix_cleaned.csv")


def process_series_matrix(file_path):
    print(f"🚀 开始处理文件: {file_path}")

    if not os.path.exists(file_path):
        print("❌ 错误: 文件不存在，请检查路径！")
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
            print("❌ 未能在文件头部找到样本信息 (!Sample_geo_accession 或 !Sample_title)")
            return

        clean_titles = []
        for t in titles:
            t = t.strip()

            if "lupus nephritis" in t:
                new_name = t.replace("SLE patient with lupus nephritis", "SLE+LN")

            elif "SLE patient" in t:
                new_name = t.replace("SLE patient", "SLE")

            elif "normal control" in t:
                new_name = t.replace("normal control", "NC")

            else:
                new_name = t

            new_name = new_name.replace("-", "").replace(" ", "")

            clean_titles.append(new_name)

        sample_map = dict(zip(accessions, clean_titles))
        print(f"✅ 成功提取 {len(sample_map)} 个样本的元数据。")
        print(f"   样本示例: {list(sample_map.values())[:5]} ...")

        print("⏳ 正在读取表达矩阵 (可能需要几秒钟)...")
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
        print("=" * 40)
        print(f"🎉 处理完成！")
        print(f"📄 结果已保存至: {OUTPUT_FILE}")
        print(f"📊 矩阵维度: {df.shape} (行: 探针, 列: 样本)")
        print(f"👀 前 5 行数据预览:")
        print(df.head())
        print("=" * 40)

    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    process_series_matrix(INPUT_FILE)