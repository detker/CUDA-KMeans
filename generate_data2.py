import numpy as np

test_data = np.zeros((160, 4))
test_data[0:40, :] = 30.0
test_data[40:80, :] = 60.0
test_data[80:120, :] = 90.0
test_data[120:, :] = 120.0

with open('data_custom2.txt', 'w') as f:
    f.write(f'{test_data.shape[0]} {test_data.shape[1]} 4\n')
    for i in range(test_data.shape[0]):
        f.write(' '.join(map(str, test_data[i])) + '\n')