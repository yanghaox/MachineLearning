print(__doc__)
# 文件注释，在py中代码最上方""" """里面一般写注释，
# 系统会把文件注释自动存放在'__doc__'这个全局变量里。

"""
使用SVM 和 RBF kernel  

gamma 参数什么意思
我们通过公式可知，gamma 是一个常量，而且是一个线性的因数。所以大家可以想象，gamma的作用，其实就是控制数据在向高维度投影后的缩放比例。如果 gamma 很大，那么上图的点就会离切面很远。如果 gamma 很小，上图的点就会离切面很近。

而这个缩放比例就会影响线性分割面的运算结果（不同的loss function对距离的惩罚度不一样）。这也是SVM对数据 Scaling 和 Normalization 是敏感的原因之一。因为最后都是算的一个 Linear Model

这就是为什么，有人说如果原始数据比较分散，gamma可以小一点。反之，如果原始数据很密集，gamma可以大一点。当然，这不是绝对的，所以我们才要做 GridSearch

通常我们会 0.01、0.1、1、10、100 ... 这样指数级地搜索一个比较好的 gamma

 C是惩罚系数，即对误差的宽容度。c越高，说明越不能容忍出现误差,容易过拟合。
 
 C越小，容易欠拟合。C过大或过小，泛化能力变差
"""

# 导入数据集 分类器 和矩阵
# 抓取原始mnist数据
from sklearn.datasets import fetch_mldata

# from mnist_helpers import*

mnist = fetch_mldata('MNIST ORIGINAL, ', data_home='$HOME/scikit_learn_data')

images = mnist.data
targets = mnist.target
