import numpy as np

def partitions(img, tile_size, n_tiles):
    X = []
    T = []

    for _ in range(n_tiles):
        buffer = int(tile_size/2)
        x_start = np.random.randint(buffer, img.shape[0] - (tile_size + buffer))
        y_start = np.random.randint(buffer, img.shape[1] - (tile_size + buffer))

        X.append(img[x_start:x_start + tile_size, y_start:y_start + tile_size])
        x_start_t = x_start - buffer
        y_start_t = y_start - buffer
        T.append(img[x_start_t:x_start + tile_size + buffer, y_start_t:y_start + tile_size + buffer])

    return X, T

def images(X, tile_size, n_tiles):
    X2= []
    T2 = []

    for d in X:
        x, t = partitions(d, tile_size, n_tiles)
        X2.append(np.array([np.array(x1).flatten() for x1 in x]))
        T2.append(np.array([np.array(t1).flatten() for t1 in t]))

    X2 = np.array(X2)
    T2 = np.array(T2)

    Xf = []
    Tf = []

    for i in range(X2.shape[0]):
        x = X2[i, :, :]
        t = T2[i, :, :]

        for i in range(x.shape[0]):
            Xf.append(x[i, :])
            Tf.append(t[i, :])

    Xf = np.array(Xf)
    Tf = np.array(Tf)
    
    return Xf, Tf

