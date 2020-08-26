import numpy as np

def cm(data):
    cx = 0 
    cy = 0
    result = np.zeros(data.shape)
    N = len(data)
    for i in range(0, N):
        cx += data[i][0]
        cy += data[i][1]

    cx /=N
    cy /=N
    print(cx)
    print(cy)
    for i in range(0, N):
        result[i][0]=data[i][0]-cx
        result[i][1]=data[i][1]-cy

    return cx,cy, result

def scaling(data):
    sx = 0
    sy = 0
    result = np.zeros(data.shape)
    N = len(data)

    for i in range(0, N):
        sx += abs(data[i][0])
        sy += abs(data[i][1])

    sx /= N
    sy /= N
    for i in range(0, N):
        result[i][0]=data[i][0]/sx
        result[i][1]=data[i][1]/sy

    return sx, sy, result 

def scaling2(data):
    dist = 0
    result = np.zeros(data.shape)
    N = len(data)

    for i in range(0, N):
        dist += np.sqrt(data[i][0]**2+data[i][1]**2)

    data /= dist/N

    return data;


data = np.random.randint(0,200,(100,2))

data=data.astype(np.float)

cx, cy, data_cm = cm(data.copy())

sx, sy, data_no = scaling(data_cm.copy())
sx, sy, data_no2 = scaling(data_cm.copy())

#print(data_cm)
print(np.sum(data_no))
print(np.sum(data_no2))

