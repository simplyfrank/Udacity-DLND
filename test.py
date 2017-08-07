import numpy as np
labels = ['snake', 'todd', 'rattle']

dumms = []
for each in labels:
    dummies = [int(each==y) for y in labels]
    dumms.append(dummies)
print(np.array(set(dumms)).shape)
