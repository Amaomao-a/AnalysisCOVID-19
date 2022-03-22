import pandas as pd
import numpy as np
import matplotlib
import csv
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# part1.数据预处理
df = []  # 将15张表的数据全部载入一个列表中，df[i]表示series对象
world = []  # 单独存储全球疫情数据用于观测全球的趋势
for i in range(4, 18 + 1):  # 打开15天的数据文件加入列表
    if i < 10:
        fileNameStr = ('Data' + '2020-12-0{}' + '.csv').format(i)
    else:
        fileNameStr = ('Data' + '2020-12-{}' + '.csv').format(i)
    df.append(pd.read_csv(fileNameStr, encoding='utf-8'))

    world_new_confirm = sum(df[i - 4]['new_confirm'])
    world_total_confirm = sum(df[i - 4]['total_confirm'])
    world_heal = sum(df[i - 4]['heal'])
    world_dead = sum(df[i - 4]['dead'])
    data = ['全球', world_new_confirm, world_total_confirm, world_heal, world_dead]
    world.append(data)

length = len(df)

country_dict = {}  # 将国家与人口存储为字典方便查询
# 同时适当人工补全数据中部分缺失的国家（有的国家有疫情数据，但没有人口数据以及命名差异带来的谬误（刚果布与刚果金））
with open('countryMsg.csv', 'r', encoding='utf-8') as csvfile:
    f_csv = csv.reader(csvfile)
    for row in f_csv:
        country_dict[row[1]] = row[2].replace(',', '')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加对中文字体的支持

# part2.数据分析与展示

# a)15天中，全球新冠疫情的总体变化趋势(日新增确诊/累计确诊/累计治愈/累计死亡)
fig1 = plt.figure()
# a1.日新增确诊折线图
ax1 = plt.subplot(2, 2, 1)
x = np.linspace(1, length, length)  # 在1-15区间内生成15个数据
ax1.set_xticks(np.arange(1, length + 1))  # 设置x轴的刻度
data = []  # 提取出日新增数据
for i in world:
    data.append(i[1])

plt.plot(x, np.array(data), color='blue', linewidth=2, linestyle='-', label='15天中全球日新增确诊变化曲线')
plt.legend(loc='lower right')
plt.title(label='15天中全球日新增确诊变化曲线')

# a2.累计确诊折线图
ax2 = plt.subplot(2, 2, 2)
x = np.linspace(1, length, length)  # 在1-15区间内生成15个数据
ax2.set_xticks(np.arange(1, length + 1))  # 设置x轴的刻度
data = []  # 提取出累计确诊数据
for i in world:
    data.append(i[2])

plt.plot(x, np.array(data), color='red', linewidth=2, linestyle='-', label='15天中全球累计确诊变化曲线')
plt.legend(loc='lower right')
plt.title(label='15天中全球累计确诊变化曲线')

# a3.累计治愈折线图
ax3 = plt.subplot(2, 2, 3)
x = np.linspace(1, length, length)  # 在1-15区间内生成15个数据
ax3.set_xticks(np.arange(1, length + 1))  # 设置x轴的刻度
data = []  # 提取出累计治愈数据
for i in world:
    data.append(i[3])

plt.plot(x, np.array(data), color='yellow', linewidth=2, linestyle='-', label='15天中全球累计治愈变化曲线')
plt.legend(loc='lower right')
plt.title(label='15天中全球累计治愈变化曲线')

# a4.累计死亡折线图
ax4 = plt.subplot(2, 2, 4)
x = np.linspace(1, length, length)  # 在1-15区间内生成15个数据
ax3.set_xticks(np.arange(1, length + 1))  # 设置x轴的刻度
data = []  # 提取出累计死亡数据
for i in world:
    data.append(i[4])

plt.plot(x, np.array(data), color='green', linewidth=2, linestyle='-', label='15天中全球累计死亡变化曲线')
plt.legend(loc='lower right')
plt.title(label='15天中全球累计死亡变化曲线')

fig1.savefig('a_15天中全球新冠疫情的总体变化趋势.png')

# b)累计确诊数排名前20的国家名称及其数量
df[length - 1].sort_values('total_confirm', ascending=False)  # 取第15天的累计确诊数据作为画图依据
x = []
y = []
for i in range(0, 20):
    x.append(df[length - 1]['country'][i])
    y.append(df[length - 1]['total_confirm'][i])

fig2, ax = plt.subplots()
color_list = ['red', 'blue', 'green', 'purple', 'pink', 'brown', 'orange']  # 用于直方图中不同国家的区分
plt.bar(x, y, width=0.4, alpha=0.5, color=color_list)
for a, b in zip(x, y):
    plt.text(a, b + 0.2, '%d' % b, ha='center', va='bottom', fontsize=10)
plt.title('累计确诊排名前20的国家')
print("\n====== 累计确诊前20的国家 ======")
for i in range(0, 20):
    print(i + 1, x[i], y[i])
fig2.savefig('b_累计确诊前20的国家名称及其数量.png')

# c)15天中，每日新增确诊数排名前10的国家的日新增确诊数据曲线图
# 最后一天的累计确诊减去第一天的累计确诊即为15内的新增确诊，从中取前10做分析
data_list = []
for i in range(len(df[length-1]['total_confirm'])):
    data_list.append([df[length-1]['country'][i], df[length-1]['total_confirm'][i]-df[0]['total_confirm'][i]])
data_list.sort(key=lambda x: -x[1])

Data = []
for i in range(0, 10+1):
    data = []
    for k in range(len(df[0]['new_confirm'])):
        if df[0]['country'][k] == data_list[i][0]:
            for j in range(0, length):
                data.append(df[j]['new_confirm'][k])
            break
    Data.append([df[0]['country'][k], data])

fig4, ax4 = plt.subplots()
x = np.linspace(1, length, length)  # 在1-15区间内生成15个数据
ax4.set_xticks(np.arange(1, length + 1))  # 设置x轴的刻度

plt.plot(x, Data[0][1], color='red', linewidth=2, linestyle='-', label=Data[0][0])
plt.plot(x, Data[1][1], color='blue', linewidth=2, linestyle='--', label=Data[1][0])
plt.plot(x, Data[2][1], color='pink', linewidth=2, linestyle='-.', label=Data[2][0])
plt.plot(x, Data[3][1], color='yellow', linewidth=2, linestyle=':', label=Data[3][0])
plt.plot(x, Data[4][1], color='green', linewidth=2, linestyle='-', label=Data[4][0])
plt.plot(x, Data[5][1], color='grey', linewidth=2, linestyle='--', label=Data[5][0])
plt.plot(x, Data[6][1], color='brown', linewidth=2, linestyle='-.', label=Data[6][0])
plt.plot(x, Data[7][1], color='orange', linewidth=2, linestyle=':', label=Data[7][0])
plt.plot(x, Data[8][1], color='purple', linewidth=2, linestyle='-', label=Data[8][0])
plt.plot(x, Data[9][1], color='blue', linewidth=2, linestyle='--', label=Data[9][0])
plt.title('日新增排名前10的国家的变化趋势图')
plt.legend(loc='upper right')

fig4.savefig('c_日增确诊累计排名前10的国家的日新增变化曲线.png')

# d)累计确诊人数占总人口比例最高的10个国家(确诊/总人口)
infection_rate = []  # 取第15天的累计确诊数据作为画图依据
for i in range(len(df[length - 1]['total_confirm'])):
    nation = df[length - 1]['country'][i]
    infection_rate.append([nation, df[length-1]['total_confirm'][i] / int(country_dict[nation]), df[length-1]['total_confirm'][i], int(country_dict[nation])])
infection_rate.sort(key=lambda x: -x[1])  # 按确诊比例降序排列并输出其中最大的前十国家
print("\n====== 累计确诊人数占国家总人口比例最高的10个国家 ======")
for i in range(0, length):
    print(infection_rate[i][0], infection_rate[i][1], infection_rate[i][2], infection_rate[i][3])

# e)新冠患病致死率最低的10个国家(死亡/确诊)
death_rate = []  # 取第15天的累计死亡人数作为依据
for i in range(len(df[length - 1]['dead'])):
    nation = df[length - 1]['country'][i]
    death_rate.append([nation, df[length-1]['dead'][i] / df[length-1]['total_confirm'][i], df[length-1]['dead'][i], df[length-1]['total_confirm'][i]])
death_rate.sort(key=lambda x: x[1])  # 按死亡率升序排列并输出其中最低的30个国家
print("\n====== 死亡率最低的30个国家 ======")  # 考虑到爬取的数据中有大量的'零患死亡'国家，适当延伸，列出30个国家，使数据有分析意义
for i in range(0, 30):
    print(death_rate[i][0], death_rate[i][1], death_rate[i][2], death_rate[i][3])

# f)用饼图展示各个国家的累计确诊人数的比例（数据量较小的国家合并为其他）
fig3 = plt.figure()
plt.title('全球各国累计确诊人数所占比例的饼图')
section_label = []
section_size = []
total_num = world[length-1][2]

for i in range(1, 20+1):
    section_label.append(df[length-1]['country'][i-1])
    section_size.append(df[length-1]['total_confirm'][i-1])
    total_num -= df[length-1]['total_confirm'][i-1]
else:
    section_label.append('其他国家')
    section_size.append(total_num)

patches, texts, autotexts = plt.pie(section_size, labels=section_label, labeldistance=1.1, autopct="%1.1f%%", shadow=True, startangle=60, pctdistance=0.6)

proptease = fm.FontProperties()
proptease.set_size('small')
plt.setp(texts, fontproperties=proptease)
plt.setp(autotexts, fontproperties=proptease)
fig3.savefig('f_饼图展示全球各国累计确诊人数的比例.png')

# g)展示全球各个国家累计确诊人数的箱型图，要有平均值
fig5 = plt.figure()
ax = df[length-1].boxplot(column=['total_confirm'], meanline=True, showmeans=True, vert=True)
ax.text(1.1, df[length-1]['total_confirm'].mean(), df[length-1]['total_confirm'].mean())
ax.text(1.1, df[length-1]['total_confirm'].median(), df[length-1]['total_confirm'].median())
ax.text(0.9, df[length-1]['total_confirm'].quantile(0.25), df[length-1]['total_confirm'].quantile(0.25))
ax.text(0.9, df[length-1]['total_confirm'].quantile(0.75), df[length-1]['total_confirm'].quantile(0.75))
fig5.savefig('g_箱型图展示全球各国累计确诊人数.png')

# h)展示治愈率最高的前30个国家
heal_rate = []  # 取第15天的累计治愈人数作为依据
for i in range(len(df[length - 1]['heal'])):
    nation = df[length - 1]['country'][i]
    heal_rate.append([nation, df[length-1]['heal'][i] / df[length-1]['total_confirm'][i], df[length-1]['heal'][i], df[length-1]['total_confirm'][i]])
heal_rate.sort(key=lambda x: -x[1])  # 按治愈率升序排列并输出其中最低的30个国家
print("\n====== 治愈率最高的30个国家 ======")
for i in range(0, 30):
    print(heal_rate[i][0], heal_rate[i][1], heal_rate[i][2], heal_rate[i][3])

# part3.数据预测: 针对全球累计确诊数，利用前10天的数据做后5天的预测，并与实际数据进行对比
world_data = []
for i in world:
    world_data.append(i[2])
world_data = DataFrame({'total_confirm':world_data})

fig6, ax = plt.subplots()
# 归一化
scaler = MinMaxScaler()
x_reshape = world_data['total_confirm'].values.reshape(-1,1)
total_confirm = scaler.fit_transform(x_reshape)

# 生成滑动窗口为2的预测值
predict_2 = world_data['total_confirm'].rolling(window=2, center=False).mean()
print("\n2-days mean error:", (world_data['total_confirm'] - predict_2).mean())                  # 平均误差
print("2-days mean absolute deviation:", abs(world_data['total_confirm'] - predict_2).mean())  # 平均绝对误差
print("2-days standard deviation:", (world_data['total_confirm'] - predict_2).std(), '\n')     # 均方误差

# 生成滑动窗口为3的预测值
predict_3 = world_data['total_confirm'].rolling(window=3, center=False).mean()
print("3-days mean error:", (world_data['total_confirm'] - predict_3).mean())
print("3-days mean absolute deviation:", abs(world_data['total_confirm'] - predict_3).mean())
print("3-days standard deviation:", (world_data['total_confirm'] - predict_3).std(), '\n')

# 生成滑动窗口为5的预测值
predict_5 = world_data['total_confirm'].rolling(window=5, center=False).mean()
print("5-days mean error:", (world_data['total_confirm'] - predict_5).mean())
print("5-days mean absolute deviation:", abs(world_data['total_confirm'] - predict_5).mean())
print("5-days standard deviation:", (world_data['total_confirm'] - predict_5).std(), '\n')

# 生成滑动窗口为10的预测值
predict_10 = world_data['total_confirm'].rolling(window=10, center=False).mean()
print("10-days mean error:", (world_data['total_confirm'] - predict_10).mean())
print("10-days mean absolute deviation:", abs(world_data['total_confirm'] - predict_10).mean())
print("10-days standard deviation:", (world_data['total_confirm'] - predict_10).std(), '\n')

ax.plot(np.arange(15), world_data['total_confirm'],color='purple', marker='o')
ax.plot(np.arange(15), predict_2, color='green', marker='o')
ax.plot(np.arange(15), predict_3, color='blue', marker='o')
ax.plot(np.arange(15), predict_5, color='orange', marker='o')
ax.plot(np.arange(15), predict_10, color='yellow', marker='o')
ax.legend(["Orig", "2-days", "3-days", "5-days", "10-days"])

arima = ARIMA(world_data['total_confirm'], order=(2, 0, 0))
result = arima.fit(disp=False)
print(result.aic, result.bic, result.hqic)
plt.plot(result.fittedvalues + 72.4, color='red')
fig6.savefig('数据预测.png')

plt.show()