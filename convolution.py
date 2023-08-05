import numpy as np

HORIZONTAL_KERNEL = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
VERTICAL_KERNEL = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

def dot_prod(m1, m2):
    assert m1.shape == m2.shape
    return np.sum(m1.flatten() * m2.flatten())

def convolve(img, ker) -> np.ndarray:
    imshape = img.shape
    assert ker.shape[0] == ker.shape[1]
    offset = ker.shape[0] - 2
    out = np.ndarray((img.shape[0] - (2 * offset), img.shape[1] - (2 * offset)))

    for x_pos in range(out.shape[0]-offset):
        for y_pos in range(out.shape[1]-offset):
            img_x_pos = x_pos
            img_y_pos = y_pos
            slice = img[img_x_pos:img_x_pos + ker.shape[0], img_y_pos:img_y_pos + ker.shape[1]]
            out[x_pos, y_pos] = dot_prod(slice, ker)
            
    
    return out