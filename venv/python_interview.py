

'''
a = 1
a = 2
def fun(a):
	a = 3
fun(a)
print(a)
#1
'''


'''
a = []
def fun(a):
	a.append(1)
fun(a)
print(a)
'''

'''
a = 1
def fun(a):
	print("func in", id(a))
	a = 2
	print("re-point", id(a), id(2))
	#fun(a) recursion Error.
	print(a)
print("func_out", id(a),id(1))
fun(a)
print(a)
print("func_out", id(a),id(1))
'''

'''
a = []
def fun(a):
	print ("fun_in", id(a))
	a.append(1)
print("func_out",id(a))
fun(a)
print(a)

'''

'''
conclusion above all: 
python中，strings, tuples, 和numbers是不可更改的对象，而 list, dict, set 等则是可以修改的对象。
“可更改”（mutable）与“不可更改”（immutable）
'''

def foo(x):
	print("execution foo(%s)" %(x))
