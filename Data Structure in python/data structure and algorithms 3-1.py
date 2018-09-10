"""
queque
入队为 $$O(n)$$，出队为 $$O(1)$$。
class Queue:
	def __init__(self):
		self.items = []

	def isEmpty(self):
		return self.items == []

	def enqueue(self, item):
		self.items.insert(0, item)

	def dequeque(self):
		return self.items.pop()

	def size(self):
		return len(self.items)
"""

"""
烫手山芋（击鼓传花） 或者约瑟夫斯问题
我们的程序将输入名称列表和一个称为 num 常量用于报数。
它将返回以 num 为单位重复报数后剩余的最后一个人的姓名。
"""
"""
['Bill']
			['David', 'Bill']
			['Susan', 'David', 'Bill']
			['Jane', 'Susan', 'David', 'Bill']
			['Kent', 'Jane', 'Susan', 'David', 'Bill']
			['Brad', 'Kent', 'Jane', 'Susan', 'David', 'Bill']
			
			0 ['Bill', 'Brad', 'Kent', 'Jane', 'Susan', 'David']
		1 ['David', 'Bill', 'Brad', 'Kent', 'Jane', 'Susan']
		2 ['Susan', 'David', 'Bill', 'Brad', 'Kent', 'Jane']
		after delete:  ['Susan', 'David', 'Bill', 'Brad', 'Kent']
		0 ['Kent', 'Susan', 'David', 'Bill', 'Brad']
		1 ['Brad', 'Kent', 'Susan', 'David', 'Bill']
		2 ['Bill', 'Brad', 'Kent', 'Susan', 'David']
		after delete:  ['Bill', 'Brad', 'Kent', 'Susan']
		0 ['Susan', 'Bill', 'Brad', 'Kent']
		1 ['Kent', 'Susan', 'Bill', 'Brad']
		2 ['Brad', 'Kent', 'Susan', 'Bill']
		after delete:  ['Brad', 'Kent', 'Susan']
		0 ['Susan', 'Brad', 'Kent']
		1 ['Kent', 'Susan', 'Brad']
		2 ['Brad', 'Kent', 'Susan']
		after delete:  ['Brad', 'Kent']
		0 ['Kent', 'Brad']
		1 ['Brad', 'Kent']
		2 ['Kent', 'Brad']
		after delete:  ['Kent']
		Kent
		
def hotPotato(namelist, num):
	simqueue = Queue()

	for name in namelist:
		simqueue.enqueue(name)
		print(simqueue.items)
			


	#print(simqueue.dequeue())
	while simqueue.size() > 1:
		for i in range(num):

			simqueue.enqueue(simqueue.dequeue())
			#print(i, simqueue.items)
		simqueue.dequeue()
		#print("after delete: ",simqueue.items)
		
		
		
	return simqueue.dequeue()

print(hotPotato(["Bill","David","Susan","Jane","Kent","Brad"],3))
	"""
"""
n 个元素的数组向右旋转 k 步。


def rotate(num, times):
	numList = Queue()

	for number in num:
		numList.enqueue(number)
		print(numList.items)

	numList.items.sort()
	for i in range(times):
		numList.enqueue(numList.dequeue())

	return numList.items

print(rotate([1,2,3,4,5,6], 3))

"""
