# 实际需要得到[3, 3]
dy = [2, 1, 3, 3, 3, 2, 2, 2]
dx = [1, 4, 3, 3, 3, 2, 5, 2]

# zip 函数接受任意多个可迭代对象作为参数,将对象中对应的元素打包成一个tuple
zipped = zip(dx, dy)

zip_list = list(zipped)

zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))
#
# print(zip_dict)
# print(zip_dict.items())
# print(zip_dict_sorted)

dxMode = list(zip_dict_sorted)[0][0]
dyMode = list(zip_dict_sorted)[0][1]

count = zip_dict_sorted[list(zip_dict_sorted)[0]]
#
print("dx = " + str(dxMode))
print("dy = " + str(dyMode))
print("count = " + str(count))