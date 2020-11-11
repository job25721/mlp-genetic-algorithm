# import numpy as np
# x = [1, 2, 3, 4, 5]
# x.reverse()

# a = [x.pop() for i in range(2)]
# # print(a)


# mydict = [{
#     "err": 0.12,
#     "p": 'sth'
# }, {"err": 0.01, "p": "sth"}]

# mydict.sort()
# sorted(mydict.index)
# print(mydict)
# for i, val in enumerate(["A", "B", "C"]):
#     # print(mydict.keys(f'{i}'))
#     pass
#     # mydict.update({f'{i}': []})
#     # mydict[f'{i}'].append(val)

#     # print(mydict)
import numpy as np
c = 0
for i in range(100000):
    x = np.random.randint(0, 1)
    if x == 1:
        c += 1


print(c)
