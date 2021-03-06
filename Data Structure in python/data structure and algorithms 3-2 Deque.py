"""
Deque
在 removeFront 中，我们使用 pop 方法从列表中删除最后一个元素。
但是，在removeRear中，pop(0)方法必须删除列表的第一个元素。
同样，我们需要在 addRear 中使用insert方法（第12行），
因为 append 方法在列表的末尾添加一个新元素。

你可以看到许多与栈和队列中描述的 Python 代码相似之处。
你也可能观察到，在这个实现中，从前面添加和删除项是 $$O(1)$$，
而从后面添加和删除是 $$O(n)$$。
考虑到添加和删除项是出现的常见操作，这是可预期的。
同样，重要的是要确定我们知道在实现中前后都分配在哪里。


from pythonds.basic.deque import Deque

def palchecker(aString):
	chardeque = Deque()

	for ch in aString:
		chardeque.addRear(ch)
		print(chardeque.items)
	stillEqual = True

	while chardeque.size() > 1 and stillEqual:
		first = chardeque.removeFront()
		print("first", first)
		last = chardeque.removeRear()
		print("last", last)
		if first != last:
			stillEqual = False

	return stillEqual

print(palchecker("lkdkjfskl"))
#print(palchecker("radar"))

"""
