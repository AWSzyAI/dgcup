import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

filename1 = 'data/附件1：各园区典型日负荷数据.xlsx'
filename2 = 'data/附件2：各园区典型日风光发电数据改.xlsx'
filename3 = 'data/附件3：12个月各园区典型日风光发电数据.xlsx'

# 读取Excel文件
df1 = pd.read_excel(filename1)
df2 = pd.read_excel(filename2, skiprows=1)
df3 = pd.read_excel(filename3, skiprows=3, header=None)

# 装机总量
P_pv_A = 750
P_w_B = 1000
P_pv_C = 600
P_w_C = 500
total_capacity = [P_pv_A, P_w_B, P_pv_C, P_w_C]

# 计算各园区日购电量总和
column_sums = df1.iloc[:, 1:].sum()
new_column_names = ["园区A(kWh)", "园区B(kWh)", "园区C(kWh)"]
column_sums_df = pd.DataFrame(column_sums.values, index=new_column_names, columns=['Sum'])

# 计算发电量
data_columns = df2.columns[1:]
Gen = df2.copy()
for i, col in enumerate(data_columns):
    Gen[col] = df2[col] * total_capacity[i]

Gen_array = Gen.iloc[:, 1:].to_numpy()
Gen_array[:, -2] += Gen_array[:, -1]
new_column = Gen_array[:, -2] - Gen_array[:, -1]
Gen_array = np.column_stack((Gen_array, new_column))

# 计算用电量
Consume_array = df1.iloc[:, 1:4].to_numpy()

# 分割矩阵
print(Gen_array)
Gen_A = Gen_array[:, 0]
Gen_B = Gen_array[:, 1]
Gen_C_W = Gen_array[:, 3]
Gen_C_L = Gen_array[:, 4]

Consume_A = Consume_array[:, 0]
Consume_B = Consume_array[:, 1]
Consume_C = Consume_array[:, 2]

# 计算弃电量
curt_A = Gen_A - Consume_A
curt_B = Gen_B - Consume_B
curt_C = np.zeros_like(Gen_C_L)
curt_C_L = np.zeros_like(Gen_C_L)
curt_C_W = np.zeros_like(Gen_C_L)

for i in range(Gen_array.shape[0]):
    if Gen_C_L[i] + Gen_C_W[i]<= Consume_C[i]:
        curt_C[i] = Gen_C_L[i]+Gen_C_W[i]-Consume_C[i]
        curt_C_L[i] = 0
        curt_C_W[i] = 0
    else:
        if Gen_C_L[i] - Consume_C[i] >= 0:
            curt_C[i] = Gen_C_L[i] - Consume_C[i] + Gen_C_W[i]
            curt_C_L[i] = Gen_C_L[i] - Consume_C[i]
            curt_C_W[i] = Gen_C_W[i]
        else:
            curt_C_L[i] = 0
            curt_C_W[i] = Gen_C_W[i] - (Consume_C[i] - Gen_C_L[i])
            curt_C[i] = Gen_C_L[i] + Gen_C_W[i] - Consume_C[i]

curt_A0 = curt_A.copy()
curt_B0 = curt_B.copy()
curt_C0 = curt_C.copy()
curt_C_L0 = curt_C_L.copy()
curt_C_W0 = curt_C_W.copy()

curt_A0[curt_A0 <= 0] = 0
curt_B0[curt_B0 <= 0] = 0
curt_C0[curt_C0 <= 0] = 0
curt_C_L0[curt_C_L0 <= 0] = 0
curt_C_W0[curt_C_W0 <= 0] = 0

# 成本计算
cost_A = np.zeros_like(Gen_C_L)
cost_B = np.zeros_like(Gen_C_L)
cost_C = np.zeros_like(Gen_C_L)


Buy_A_WANG = np.zeros_like(Gen_C_L)

for i in range(len(curt_A)):
    if curt_A[i] < 0:
        Buy_A_WANG[i] =  (-curt_A[i])
        cost_A[i] = 1 * (-curt_A[i]) + 0.4 * Gen_A[i]
    else:
        Buy_A_WANG[i] = 0
        cost_A[i] = 0.4 * Consume_A[i]

Buy_B_WANG = np.zeros_like(Gen_C_L)

for i in range(len(curt_B)):
    if curt_B[i] < 0:
        Buy_B_WANG[i] =  (-curt_B[i])
        cost_B[i] = 1 * (-curt_B[i]) + 0.5 * Gen_B[i]
    else:
        Buy_B_WANG[i] = 0
        cost_B[i] = 0.5 * Consume_B[i]

Buy_C_L = np.zeros_like(Gen_C_L)
Buy_C_W = np.zeros_like(Gen_C_L)
Buy_C_WANG = np.zeros_like(Gen_C_L)

print(Gen_C_L)

for i in range(len(curt_C)):
    if curt_C[i] < 0:
        cost_C[i] = 1 * (-curt_C[i]) + 0.4 * Gen_C_L[i] + 0.5 * Gen_C_W[i]
        Buy_C_WANG[i] =  (-curt_C[i])
        Buy_C_L[i] = Gen_C_L[i]
        Buy_C_W[i] = Gen_C_W[i]
    else:
        Buy_C_WANG[i] =0
        if Gen_C_L[i] >= Consume_C[i]:
            cost_C[i] = 0.4 * Consume_C[i]
            Buy_C_L[i] = Consume_C[i]
            Buy_C_W[i] = 0
        else:
            cost_C[i] = 0.4 * Gen_C_L[i] + 0.5 * (Consume_C[i]-Gen_C_L[i])
            Buy_C_L[i] = Gen_C_L[i]
            Buy_C_W[i] = Consume_C[i]-Gen_C_L[i]

#保存园区A的用电量、光伏供电量到csv文件
df_A = pd.DataFrame({
    '用电量': Consume_A,
    '光伏供电量': Gen_A,
})
df_A.to_csv('problem1/园区A.csv', index=False)

#保存园区B的用电量、风电供电量到csv文件
df_B = pd.DataFrame({
    '用电量': Consume_B,
    '风电供电量': Gen_B,
})
df_B.to_csv('problem1/园区B.csv', index=False)

#保存园区C的用电量、光伏供电量、风电供电量到csv文件
df_C = pd.DataFrame({
    '用电量': Consume_C,
    '光伏供电量': Gen_C_L,
    '风电供电量': Gen_C_W,
})
df_C.to_csv('problem1/园区C.csv', index=False)

# 打印结果
print("园区A总用电量:", sum(Consume_A))
print("园区A网购购电量:", sum(Buy_A_WANG))
print("园区B总用电量:", sum(Consume_B))
print("园区B网购购电量:", sum(Buy_B_WANG) )
print("园区C总用电量:", sum(Consume_C))
print("园区C网购购电量:", sum(Buy_C_WANG))
print("园区C光伏购电量:", sum(Buy_C_L) )
print("园区C风电购电量:", sum(Buy_C_W) )

print("园区A每天弃电量:", sum(curt_A0))
print("园区B每天弃电量:", sum(curt_B0))
print("园区C每天弃电量:", sum(curt_C0))
print("园区C每天光伏弃电量:", sum(curt_C_L0))
print("园区C每天风电弃电量:", sum(curt_C_W0))

print("园区A成本:", sum(cost_A))
print("园区B成本:", sum(cost_B))
print("园区C成本:", sum(cost_C))

print("园区A总发电量:", sum(Gen_A))
print("园区B总发电量:", sum(Gen_B))
print("园区C总发电量:", sum(Gen_C_L) + sum(Gen_C_W))

print("园区A单位电量平均供电成本:")
print(sum(cost_A) / sum(Consume_A))
print("园区B单位电量平均供电成本:")
print(sum(cost_B) / sum(Consume_B))
print("园区C单位电量平均供电成本:")
print(sum(cost_C) / (sum(Consume_C)))


plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] =False

# 绘制园区A用电折线图
plt.figure(figsize=(10, 6))
plt.plot(Consume_A, label='用电量', color='blue')
plt.plot(curt_A0, label='弃电量', color='green')
plt.plot(Buy_A_WANG, label='主电网购入量', color='red')
plt.plot(Consume_A-Buy_A_WANG, label='光伏购入量', color='yellow')
plt.xlabel('时间(h)')
plt.ylabel('电量(kwh)')
plt.title('A园区各电量折线图')
plt.legend()
plt.grid(True)
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = 'A园区各电量折线图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()


# 绘制园区B用电折线图
plt.figure(figsize=(10, 6))
plt.plot(Consume_B, label='用电量', color='blue')
plt.plot(curt_B0, label='弃电量', color='green')
plt.plot(Buy_B_WANG, label='主电网购入量', color='red')
plt.plot(Consume_B-Buy_B_WANG, label='风电购入量', color='yellow')
plt.xlabel('时间(h)')
plt.ylabel('电量(kwh)')
plt.title('B园区各电量折线图')
plt.legend()
plt.grid(True)
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = 'B园区各电量折线图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()

# 绘制园区C用电折线图
plt.figure(figsize=(10, 6))
plt.plot(Consume_C, label='用电量', color='blue')
plt.plot(curt_C0, label='弃电量', color='green')
plt.plot(Buy_C_WANG, label='主电网购入量', color='red')
plt.plot(Buy_C_L, label='光伏购入量', color='yellow')
plt.plot(Buy_C_W, label='风电购入量', color='purple')
plt.xlabel('时间(h)')
plt.ylabel('电量(kwh)')
plt.title('A园区各电量折线图')
plt.legend()
plt.grid(True)
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = 'C园区各电量折线图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()


# 绘制园区A,B,C成本折线图
plt.figure(figsize=(10, 6))
plt.plot(cost_A, label='园区A成本', color='blue')
plt.plot(cost_B, label='园区B成本', color='green')
plt.plot(cost_C, label='园区C成本', color='red')
plt.xlabel('时间(h)')
plt.ylabel('成本(元)')
plt.title('各园区成本折线图')
plt.legend()
plt.grid(True)
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = '各园区成本折线图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()


# 绘制所有园区成本与供需差的关系图
plt.figure(figsize=(10, 6))
plt.scatter(curt_A, cost_A, color='blue', label='园区A')
plt.scatter(curt_B, cost_B, color='green', label='园区B')
plt.scatter(curt_C, cost_C, color='red', label='园区C')
plt.xlabel('供需差')
plt.ylabel('成本')
plt.title('各园区成本与供需差关系图')
plt.legend()
plt.grid(True)
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = '各园区成本与供需差关系图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()

# 绘制所有园区发电量与用电量的关系图
plt.figure(figsize=(10, 6))
plt.scatter(Gen_A, Consume_A, color='blue', label='园区A')
plt.scatter(Gen_B, Consume_B, color='green', label='园区B')
plt.scatter(Gen_C_L + Gen_C_W, Consume_C, color='red', label='园区C')
plt.xlabel('发电量')
plt.ylabel('用电量')
plt.title('各园区发电量与用电量关系图')
plt.legend()
plt.grid(True)
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = '各园区发电量与用电量关系图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()

# 绘制相关性热力图
data_A = pd.DataFrame({
    '用电量': Consume_A,
    '弃电量': curt_A0,
    '主电网购入量': Buy_A_WANG,
    '光伏购入量': Consume_A-Buy_A_WANG,
    '成本': cost_A
})

data_B = pd.DataFrame({
    '用电量': Consume_B,
    '弃电量': curt_B0,
    '主电网购入量': Buy_B_WANG,
    '风电购入量': Consume_B-Buy_B_WANG,
    '成本': cost_B
})

data_C = pd.DataFrame({
    '用电量': Consume_C,
    '弃电量': curt_C0,
    '主电网购入量': Buy_C_WANG,
    '光伏购入量': Buy_C_L,
    '风电购入量': Buy_C_W,
    '成本': cost_B
})

# 计算相关性矩阵
corr_A = data_A.corr()
corr_B = data_B.corr()
corr_C = data_C.corr()
corr_A.to_excel("correlation_A.xlsx", index=True)  # 将相关性矩阵 A 写入 Excel 文件
corr_B.to_excel("correlation_B.xlsx", index=True)  # 将相关性矩阵 B 写入 Excel 文件
corr_C.to_excel("correlation_C.xlsx", index=True)  # 将相关性矩阵 C 写入 Excel 文件

# 绘制园区A的相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr_A, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('园区A 各属性的相关性热力图')
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = '园区A 各属性的相关性热力图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()

# 绘制园区B的相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr_B, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('园区B 各属性的相关性热力图')
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = '园区B 各属性的相关性热力图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()

# 绘制园区C的相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(corr_C, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('园区C 各属性的相关性热力图')
#plt.show()
# 保存图形到指定文件夹和文件名
output_folder = 'problem1/pictures'
output_filename = '园区C 各属性的相关性热力图.png'
output_path = f'{output_folder}/{output_filename}'
plt.savefig(output_path, dpi=300)
plt.close()

