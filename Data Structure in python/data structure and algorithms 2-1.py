"""
计算前 n 个整数的和

import time
def sumOfN(n):
	start_time = time.time()
	the_sum = 0
	for i in range(1,n + 1):
		the_sum += i
		#print(i)

	end_time = time.time()
	return the_sum, end_time - start_time
for i in range(5):
	print("Sum is %d required %10.7f seconds. "%sumOfN(50000))
"""

"""
Anagram 相同字母不同顺序的词 

def anagramSolution1(s1,s2):
    alist = list(s2)

    pos1 = 0
    stillOK = True

    while pos1 < len(s1) and stillOK:
        pos2 = 0
        found = False
        print("First loop, pos1 %d"%pos1)
        while pos2 < len(alist) and not found:
            if s1[pos1] == alist[pos2]:
                found = True
                print("Second loop if, pos1 = %d pos2 = %d " %(pos1, pos2))
            else:
                pos2 = pos2 + 1
                print("Second loop else,  pos1 = %d pos2 = %d " %(pos1, pos2) )
        if found:
            alist[pos2] = None
            print("the alist is ", alist)
        else:
            stillOK = False

        pos1 = pos1 + 1

    return stillOK

print(anagramSolution1('abcqwed','qweadcb'))

#Big o is n*n
"""

"""
排列字符串 再进行比对
因为只有一个简单的迭代来比较排序后的 n 个字符。
但是，调用 Python 排序不是没有成本。
正如我们将在后面的章节中看到的，排序通常是 $$O(n^2)$$ 或 $$O(nlogn)$$。
所以排序操作比迭代花费更多。最后该算法跟排序过程有同样的量级。

def anagram2(s1,s2):
	alist = list(s1)
	blist = list(s2)

	alist.sort()
	blist.sort()

	pos = 0
	check = True
	while pos <len(alist) and check:
		if alist[pos] == blist[pos]:
			pos += 1

		else:
			return not check
	return check
print(anagram2('qwer','rewq'))
"""
"""
利用计数器比较  
我们首先计算的是每个字母出现的次数。
由于有 26 个可能的字符，我们就用 一个长度为 26 的列表，每个可能的字符占一个位置。
每次看到一个特定的字符，就增加该位置的计数器。
最后如果两个列表的计数器一样，则字符串为乱序字符串。
o(n)

虽然最后一个方案在线性时间执行，但它需要额外的存储来保存两个字符计数列表。
换句话说，该算法牺牲了空间以获得时间。


def anargram3(s1,s2):
	c1 = [0] *26
	#print(c1)  #[0, 0, 0, 0, 0, 0, ...., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	c2 = [0] * 26
	#print("ord(a), ord(s1[0])",ord('a'),ord(s1[0]) )
	#ord(a), ord(s1[0]) 97 113

	for i in range(len(s1)):
		pos = ord(s1[i]) - ord('a')
		c1[pos] = c1[pos] + 1
	for i in range(len(s2)):
		pos = ord(s2[i]) - ord('a')
		c2[pos] = c2[pos] + 1
	print(c1 ,'\n', c2 )

	j = 0
	check = True
	while j < 26 and check:
		if c1[j] == c2[j]:
			j += 1
		else:
			check = False

	print(check)


anargram3('qwertyuioppp','poiuytrewqpp')
"""

"""
我们将确认列表的 contains 操作符是 $$O(n)$$，字典的 contains 操作符是 $$O(1)$$。
我们将在实验中列出一系列数字。然后随机选择数字，并检查数字是否在列表中。

import timeit
import random

for i in range(10000,1000001,20000):
    t = timeit.Timer("random.randrange(%d) in x"%i,
                     "from __main__ import random,x")
    x = list(range(i))
    lst_time = t.timeit(number=1000)
    x = {j:None for j in range(i)}
    d_time = t.timeit(number=1000)
    print("%d,%10.3f,%10.3f" % (i, lst_time, d_time))

"""








