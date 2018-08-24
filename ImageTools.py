import numpy as np

# From: https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
def getInterpolatedPixelValues(image, x, y):
	x = np.asarray(x)
	y = np.asarray(y)
	
	x0 = np.floor(x).astype(int)
	x1 = x0 + 1
	y0 = np.floor(y).astype(int)
	y1 = y0 + 1
	
	x0 = np.clip(x0, 0, image.shape[1]-1)
	x1 = np.clip(x1, 0, image.shape[1]-1)
	y0 = np.clip(y0, 0, image.shape[0]-1)
	y1 = np.clip(y1, 0, image.shape[0]-1)
	
	center = image[y0, x0]
	top = image[y1, x0]
	right = image[y0, x1]
	topright = image[y1, x1]
	
	w_center = (x1-x) * (y1-y)
	w_top = (x1-x) * (y-y0)
	w_right = (x-x0) * (y1-y)
	w_topright = (x-x0) * (y-y0)
	
	return w_center*center + w_top*top + w_right*right + w_topright*topright

def setInterpolatedPixelValue(image, x, y, v):
    if (x <= -1) or (x > image.shape[1]):
        return
    if (y <= -1) or (y > image.shape[0]):
        return

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    
    x0 = np.clip(x0, 0, image.shape[1]-1)
    x1 = np.clip(x1, 0, image.shape[1]-1)
    y0 = np.clip(y0, 0, image.shape[0]-1)
    y1 = np.clip(y1, 0, image.shape[0]-1)
    
    center = image[y0, x0]
    top = image[y1, x0]
    right = image[y0, x1]
    topright = image[y1, x1]
    
    w_center = (x1-x) * (y1-y)
    w_top = (x1-x) * (y-y0)
    w_right = (x-x0) * (y1-y)
    w_topright = (x-x0) * (y-y0)
    
    image[y0, x0] = (1.0-w_center)*center + w_center*v
    image[y1, x0] = (1.0-w_top)*top + w_top*v
    image[y0, x1] = (1.0-w_right)*right + w_right*v
    image[y1, x1] = (1.0-w_topright)*topright + w_topright*v
