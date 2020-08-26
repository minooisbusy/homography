import numpy as np

def cm(data):
    cx = 0 
    cy = 0
    N = len(data)
    for i in range(0, N):
        cx += data[i][0]
        cy += data[i][1]

    data[:][0]-=cx/N
    data[:][1]-=cy/N

    return cx/N,cy/N, data

def scaling(data):
    sx = 0
    sy = 0
    N = len(data)

    for i in range(0, N):
        sx += abs(data[i][0])
        sy += abs(data[i][1])

    data[:][0] /= sx/N
    data[:][1] /= sy/N

    return sx/N, sy/N, data

def scaling2(data):
    dist = 0
    N = len(data)

    for i in range(0, N):
        dist += np.sqrt(data[i][0]**2+data[i][1]**2)

    data /= dist/N

    return data;


data = np.random.randint(0,200,(100,2))

data=data.astype(np.float)

cx, cy, data_cm = cm(data)
sx, sy, data_no = scaling(data_cm)

sx, sy, data_no2 = scaling(data_cm)

print(np.sum(np.abs(data_no-data_no2)))


#print(data)



