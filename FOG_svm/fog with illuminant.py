# Author: Haoxiang
# log1: xFOG was picked up from the real data(yellow color); And also Illuminant
# 导入库
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVR  # SVM中的回归算法

train_xFog = pd.DataFrame(pd.read_csv('data.csv'))

test_yFog = pd.DataFrame(pd.read_csv('yfog_guang.csv'))

'''
#plt.plot(train_xFog.wd,train_xFog.feixianxing, train_xFog.guanggonglv, 'g.','r.','y.')
pd.scatter_matrix(train_xFog)
plt.title('train_xFog')
#print(train_xFog['guanggonglv'].corr(train_xFog['wd']))
plt.show()

plt.plot(train_xFog.wending, 'g.')
plt.show()
plt.plot(train_xFog.guanggonglv, 'r.')
plt.show()
plt.plot(train_xFog.guangpu, 'y.')
plt.show()
print(train_xFog.shape)
print(test_yFog.shape)
'''

X = train_xFog.iloc[:, :6]
print(X.shape)

y = test_yFog.iloc[:, -1]
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train))
print(len(X_test))
svr = SVR()
poly_svr = SVR(kernel='poly')
svr.fit(X_train, y_train)
poly_svr.fit(X_train, y_train)
pre = svr.predict(X_test)
# y_pred = SVR.predict(X_test)
svr.score(X_test, y_test)
print('\n \n \n Default Accuracy Score:')
# print(poly_svr.score(X_test,y_test))
print(svr.score(X_test, y_test))
# plt.hist(train_xFog.wending, range=(0,1), bins=20, rwidth=0.9)
# 参数range 表示想要展示数据的区间.原来数据表信息不会修改到.在区间范围外的数据只是隐藏起来了.如(60,80)则只显示60~80岁的数据.
# 参数bins 表示将数据平均分成多少份来展示. 如20,则表示将X轴上数据平均分成20份.即显示20条柱状.
# 参数rwidth (取值范围1~0). '0.9'表示设置单条柱状的宽度显示占比90%.那么就有10%是空白的,呈现出来的直方图就美观一些.

# 显示图表出来的命令, (上节课:我们是用%matplotlib inline 直接展现).这节课:是逐一显示.画完一个图就得输入一次 plt.show()
# plt.boxplot(df.life)
# plt.show()


'''

plt.show()
pd.scatter_matrix(train_xFog)

map_dict = {      
    'Asia':'red',
    'Europe':'green',
    'Africa':'blue',
    'North America':'yellow',
    'South America':'yellow',
    'Oceania':'black'
}
colors = df.continent.map(map_dict)   # 将国家按所在州对于不同的颜色
size = df.popu / 1e6 * 2
plt.scatter(x=df.gdp, y=df.life, s=size, c=colors, alpha=0.5)  # 参数c设置颜色，alpha设置透明度
plt.xscale('log')
plt.xlabel('人均GDP（美元）')
plt.ylabel('人均寿命（年）')
plt.title('全球健康和收入水平关系（2015）')
tick_val = [1000,10000,100000]
tick_lab = ['1k','10k','100k']
plt.xticks(tick_val, tick_lab)
plt.show()
'''

'''plt.scatter(train_xFog.wending, train_xFog.feixianxing)
plt.show()


一般形式：
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data
和testdata，形式为：
X_train,X_test, y_train, y_test =
cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
参数解释：
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

from sklearn.preprocessing import StandardScaler
 sc = StandardScaler()
 sc.fit(X_train)#计算均值跟标准差
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#x = iris.data[:,[2,3]] #提取每一行中的第2，3列

'''

'''
# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
model_br = BayesianRidge()  # 建立贝叶斯岭回归模型对象
model_lr = LinearRegression()  # 建立普通线性回归模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR()  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表

model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]  # 不同回归模型对象的集合

cv_score_list = []  # 交叉检验结果列表
pre_y_list = []  # 各个回归模型预测的y值列表
for model in model_dic:  # 读出每个回归模型对象
    scores = cross_val_score(model, train_xFog, test_yFog, cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
    cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
    pre_y_list.append(model.fit(train_xFog, test_yFog).predict(X))  # 将回归训练中得到的预测y存入列表

# 模型效果指标评估
n_samples, n_features = train_xFog.shape  # 总样本量,总特征数
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # 回归评估指标对象集
model_metrics_list = []  # 回归评估指标列表
for i in range(5):  # 循环每个模型索引
    tmp_list = []  # 每个内循环的临时结果列表
    for m in model_metrics_name:  # 循环每个指标对象
        tmp_score = m(test_yFog, pre_y_list[i])  # 计算每个回归指标结果
        tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
    model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # 建立回归指标的数据框
print ('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
print (70 * '-')  # 打印分隔线
print ('cross validation result:')  # 打印输出标题
print (df1)  # 打印输出交叉检验的数据框
print (70 * '-')  # 打印分隔线
print ('regression metrics:')  # 打印输出标题
print (df2)  # 打印输出回归指标的数据框
print (70 * '-')  # 打印分隔线
print ('short name \t full name')  # 打印输出缩写和全名标题
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')  # 打印分隔线
# 模型效果可视化
plt.figure()  # 创建画布
plt.plot(np.arange(train_xFog.shape[0]), y, color='k', label='true y')  # 画出原始值的曲线
color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
    plt.plot(np.arange(train_xFog.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # 画出每条预测结果线
plt.title('regression result comparison')  # 标题
plt.legend(loc='upper right')  # 图例位置
plt.ylabel('real and predicted value')  # y轴标题
plt.show()  # 展示图像
# 模型应用
print ('regression prediction')
new_point_set = [[1.05393, 0., 8.14, 0., 0.538, 5.935, 29.3, 4.4986, 4., 307., 21., 386.85, 6.58],
                 [0.7842, 0., 8.14, 0., 0.538, 5.99, 81.7, 4.2579, 4., 307., 21., 386.75, 14.67],
                 [0.80271, 0., 8.14, 0., 0.538, 5.456, 36.6, 3.7965, 4., 307., 21., 288.99, 11.69],
                 [0.7258, 0., 8.14, 0., 0.538, 5.727, 69.5, 3.7965, 4., 307., 21., 390.95, 11.28]]  # 要预测的新数据集
for i, new_point in enumerate(new_point_set):  # 循环读出每个要预测的数据点
    new_pre_y = model_gbr.predict(new_point)  # 使用GBR进行预测
    print ('predict for new point %d is:  %.2f' % (i + 1, new_pre_y))  # 打印输出每个数据点的预测信息

'''
