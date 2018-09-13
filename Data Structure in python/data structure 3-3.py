"""
要实现 size 方法，我们需要遍历链表并对节点数计数
def size(self):
	current = self.head
	count = 0

	while current != None:
		count += 1
		current = current.getNext()

	return count

搜索方法的实现
def search(self,item):
	current = self.head
	found = False

	while current != None and not found:
		if current.getData() == item:
			found = True
		else:
			current = current.getNext()
	return found

删除方法的实现
def remove(self,item):
    current = self.head
    previous = None
    found = False
    while not found:
        if current.getData() == item:
            found = True
        else:
            previous = current
            current = current.getNext()

    if previous == None:
        self.head = current.getNext()
    else:
        previous.setNext(current.getNext())
"""

"""
十进制数转二进制数


def ToBinary(dec):
	binaryStack = Stack()

	while dec >0:
		binaryNum = dec % 2
		binaryStack.push(binaryNum)
		dec = dec // 2
	print(binaryStack.items)
	print(binaryStack.size())
	binary_string = ""
	for i in range(binaryStack.size()):
		binary_string = binary_string + str(binaryStack.pop())
		print(binary_string)

	return binary_string
print(ToBinary(20))
"""

"""
十进制转任何进制,知道转换的基数 base
def ToAny(dec,base):
	newStack = Stack()

	while dec>0:
		number = dec % base
		newStack.push(number)
		dec = dec // 2

	new_string = ""
	while not newStack.isEmpty():
		new_string = new_string + str(newStack.pop())
	return new_string

print("16进制数： ",ToAny(30,16))

"""
