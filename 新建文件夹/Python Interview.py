"""
#Python的函数参数传递
a = 1

# print("id = 1", id(1))
def fun(a):
#	print("function_id", id(a))
	a = 2
#	print("re-point", id(a), id(1), id(2))

fun(a)
#print (a)
#print("fun_out", id(a),id(1), id(2))
"python中，strings, tuples, 和numbers是不可更改的对象，" \
"而 list, dict, set 等则是可以修改的对象。" \
"这就造成了以上的变量做了改变，而其他就不会就行内存改变"

def myfunc ( n = 2):
	return lambda n : n * n
doubler = myfunc(2)
# print(myfunc(10))
# print(doubler)
"""

"interator 迭代器 generator 属于其中之一。" \
"使用array创建链表推导师 它的值在内存中 因此可以一次次的调用" \
"generator 也就是用（）创建的 只能使用一次，因为它们没有全部存在内存中" \
""
"""
mygenerrator = [x + 2 for x in range(3)]
for i in mygenerrator:
	#print("second: " ,i)
#for x in mygenerrator:
	#print(x)

def creategenerator():
	mylist = range(2)
	for i in mylist:
		yield  i*i
my = creategenerator()
for i in my:
	print(i)

"""

#计算器乘方 和 默认参数
"""
def caculator(num):
	return  num * num
print(caculator(9))
"""
"""
def caculator(num, power = 2):
	x = 1
	while power > 0:
		power = power - 1
		x = x * num
	return x
#print(caculator(5))

#可变参数 参数前面加上*就是可变参数 和 默认参数 默认参数必须指向不变对象！！！
#关键字参数  使用**表示关键字参数
#默认参数一定要用不可变对象，如果是可变对象，运行会有逻辑错误！
#*args是可变参数，args接收的是一个tuple;  **kw是关键字参数，kw接收的是一个dict
#可变对象：list,dict.不可变对象有:int,string,float,tuple
def functional(a, b, c = [1], *args, **kwargs):
	print("a = ",a , "b = ", b, "c = ", 3, "d = ", args, "e = ", kwargs)
functional(1, 'b', 5 ,"sd", 'sdef', a,a,a, D = 'dabeijing', E  = 20)
#a =  1 b =  b c =  3 d =  ('sd', 'sdef', 1, 1, 1) e =  {'D': 'dabeijing', 'E': 20}
"""

"""
#set:集合,无序,元素只出现一次, 自动去重,使用”set([])”
#list:链表,有序的项目, 通过索引进行查找,使用方括号”[]”;
#tuple:元组,元组将多样的对象集合到一起,不能修改,通过索引进行查找, 使用括号”()”;
#dict:字典,字典是一组键(key)和值(value)的组合,通过键(key)进行查找,没有顺序, 使用大括号”{}”;

myset = set([1,1,2,2,'w','w'])
print(myset)
"""
"""
#引用和copy(),deepcopy()的区别
import copy
a = [1, 2, 3, 4, ['a', 'b']]  #原始对象

b = a  #赋值，传对象的引用
c = copy.copy(a)  #对象拷贝，浅拷贝
d = copy.deepcopy(a)  #对象拷贝，深拷贝

a.append(5)  #修改对象a
a[4].append('c')  #修改对象a中的['a', 'b']数组对象

print 'a = ', a
print 'b = ', b
print 'c = ', c
print 'd = ', d

输出结果：
a =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
b =  [1, 2, 3, 4, ['a', 'b', 'c'], 5]
c =  [1, 2, 3, 4, ['a', 'b', 'c']]
d =  [1, 2, 3, 4, ['a', 'b']]
"""

