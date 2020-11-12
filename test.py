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

x = []

for i in range(4):
    r = np.random.randint(0, 5)
    while x.__contains__(r):
        r = np.random.randint(0, 5)
    x.append(r)
    print(r)
