import numpy as np
import cv2
import scipy.sparse as sp

def Laplacian(img):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian = cv2.filter2D(img, -1, kernel)
    return laplacian

def buildSystemSparse(img_shape, mask):
    h, w = img_shape
    indices = np.zeros_like(mask).astype(np.int64)
    indices[mask > 0] = np.arange(0, np.sum(mask))
    expanded_mask = np.zeros((h+2, w+2)).astype(np.bool_)
    expanded_mask[1:-1, 1:-1] = mask

    N = np.sum(mask).astype(np.int64)
    A = sp.lil_matrix((N, N))

    is_boundary = lambda x, y: mask[x, y] and (not expanded_mask[x, y+1] or not expanded_mask[x+2, y+1] or not expanded_mask[x+1, y] or not expanded_mask[x+1, y+2])
    cnt = 0
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                assert indices[i, j] == cnt
                if is_boundary(i, j):
                    A[indices[i, j], indices[i, j]] = -1
                else:
                    A[indices[i, j], indices[i, j]] = 4
                    A[indices[i, j], indices[i-1, j]] = -1
                    A[indices[i, j], indices[i+1, j]] = -1
                    A[indices[i, j], indices[i, j-1]] = -1
                    A[indices[i, j], indices[i, j+1]] = -1
                cnt += 1
    return A, indices

def computeb(fg, bg, mask, indices):
    h, w = fg.shape[:2]
    expanded_mask = np.zeros((h+2, w+2))
    expanded_mask[1:-1, 1:-1] = mask

    N = np.sum(mask).astype(np.int64)

    grad = np.dstack([Laplacian(fg[:,:,i]) for i in range(3)])

    is_boundary = lambda x, y: mask[x, y] and (not expanded_mask[x, y+1] or not expanded_mask[x+2, y+1] or not expanded_mask[x+1, y] or not expanded_mask[x+1, y+2])

    N = np.sum(mask).astype(np.int64)
    b = np.zeros((N, 3)) # rgb

    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                if is_boundary(i, j):
                    b[indices[i, j]] = np.negative(bg[i, j])
                else:
                    b[indices[i, j]] = grad[i, j]
    
    return b

def Jacobi_iteration(A, b, max_iter=1000000, tol=1e-6):
    D = A.diagonal()
    R = A - sp.diags(D, 0)
    x = np.zeros_like(b)
    for _ in range(max_iter):
        x_new = (b - R.dot(x)) / D
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x