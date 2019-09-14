import itertools

W = 64  
H = 64  
IMAGE_SHAPE = [W,H,1]
R = 3
DIRS = list(itertools.product([-R*2,0,R*2], [-R*2,0,R*2]))
DIRS.remove((0,0))
ATTR_SIZE = 7


