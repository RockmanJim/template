import pandas as pd

# 读取、分析数据集(/code/components/data)

# 模型预测

# 输出预测文件(示例)
ids = [
    '7ebfef6101d03140b3d07d550857e584.csv',
    '855e756747da36f98254c7255cd603b7.csv',
    'fdc30cbecfc533d1b13c222cd0b3508a.csv'
]
results = [0,1,0] 
dataframe = pd.DataFrame({'id':ids,'result':results})
dataframe.to_csv("test_predict.csv",index=False,sep=',')  # 将DataFrame存储为csv,index表示是否显示行名，default=True
