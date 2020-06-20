# -*- coding: utf-8 -*-
import numpy as np

random = np.random.RandomState(0)  # RandomState生成随机数种子
for i in range(200):  # 随机数个数
	# a = random.uniform(0.246, 0.561)#随机数范围, uniform：生成均匀分布
	a = np.random.normal(3.469372, 1.360432, size=None)

	print(round(a, 3))  # 随机数精度要求
