import os
import numpy as np
import pandas as pd

def convert_npy_to_excel(directory):
    # 使用 os.walk() 遍历目录及子目录
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 只处理.npy文件
            if filename.endswith('.npy'):
                # 获取文件的完整路径
                file_path = os.path.join(root, filename)
                # 读取.npy文件数据
                data = np.load(file_path)
                
                # 将数据转换为 DataFrame
                df = pd.DataFrame(data)
                
                # 生成输出的 Excel 文件路径（去掉 .npy 扩展名并添加 .xlsx 扩展名）
                output_excel = os.path.join(root, filename.replace('.npy', '.xlsx'))
                
                # 将 DataFrame 导出为 Excel 文件
                df.to_excel(output_excel, index=False, engine='openpyxl')
                print(f"已将 {filename} 转换为 Excel 文件: {output_excel}")

# 调用函数，指定目录
directory = 'test_output'  # 替换为实际的.npy文件目录

convert_npy_to_excel(directory)
