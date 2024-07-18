# importing modules 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors
from PIL import Image
import random,os

bruteforceonly = False
regionlimit = 20 # 10
bordersize = 2 # 2
mapname = "msia.jpeg"

def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns a dictionary
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image("test_images/cat.png")
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [round(.299 * p[0] + .587 * p[1] + .114 * p[2])
                      for p in img_data]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}

cur_dir = os.path.dirname(__file__) # directory of this file

x = load_greyscale_image(os.path.join(cur_dir, "maps/" + mapname))
m,n = x["height"],x["width"]
rawdata = np.array(x["pixels"]).reshape((m,n))
data = np.ones((m,n)) * 6
data[rawdata > 200] = 0

def compress(matrix, rstep, cstep):
    
    a,b = matrix.shape
    m,n = int(a/rstep), int(b/cstep)
    ans = np.zeros((m,n))
    for r in range(m):
        for c in range(n):
            local = matrix[rstep*r:rstep*(r+1),cstep*c:cstep*(c+1)]
            if np.sum(local) >= local.size *3 :
                ans[r][c] = 6
    return ans

background = compress(data, 2, 2)
m, n = background.shape

def separator(bg):
    m, n = bg.shape
    matrix = np.zeros((m,n))
    def nb(u):
        return [(u[0] + i, u[1] + j) 
                for i,j in [(-1,0),(1,0), (0,1), (0,-1)]
                if 0<=u[0]+i<m and 0<=u[1]+j<n]
    prod = [(a,b) for a in range(m) for b in range(n)]
    prod = sorted(prod, key = lambda x: x[0]+x[1])
    classify = 0
    for a,b in prod:
        if matrix[a,b] != 0 or bg[a,b] == 6: continue
        classify += 1
        size = 1
        agenda = [(a,b)]
        matrix[a,b] = classify
        while agenda:
            last = agenda.pop()
            for p,q in nb(last):
                if matrix[p,q]==0 and bg[p,q] == 0:
                    matrix[p,q] = classify
                    size+=1
                    agenda.append((p,q))
        if size < regionlimit: classify -= 1
    return matrix

newmap = separator(background)

def buckets(classified):
    m, n = classified.shape

    def nb(u, s):
        return [(u[0] + i, u[1] + j) 
                for i in range(-s,s+1)
                for j in range(-s,s+1)
                if 0<=u[0]+i<m and 0<=u[1]+j<n]

    bucket = {}

    for x in range(m):
        for y in range(n):
            classnum = int(classified[x,y])
            bucket.setdefault(classnum-2, set())
            bucket[classnum-2].add((x,y))
    
    adjpairs = set()

    for x,y in bucket[-2]:
        neighborhood = set()
        for a,b in nb((x,y), bordersize):
            neighborhood.add(int(classified[a,b])-2)
        neighborhood -= {-1}

        adjpairs |= {(a,b) for a in neighborhood for b in neighborhood if a!=b}

    adjlist = {}
    for x,y in adjpairs:
        adjlist.setdefault(x, [])
        adjlist[x].append(y)

    return bucket, adjlist

reg, adjset = buckets(newmap)

numstates = len(reg)-2 # -2 corresponds to borders, -1 corresponds to sea

# exit()

def converter(gamestate):
    addon = np.zeros((m,n))
    for id in range(numstates):
        for x,y in reg[id]: addon[x,y] = gamestate[id]
    return background+addon

cmap = colors.ListedColormap(['white', 'red', 'yellow', 'blue', 'green', 'black'])
norm = colors.BoundaryNorm([0,1,2,3,4,5,6], cmap.N)

if bruteforceonly:
    fig, ax1 = plt.subplots(1,1)
else:
    fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_figheight(3)
fig.set_figwidth(10)
fig.set_size_inches(15, 7)


fig.suptitle("4-coloring: Brute Force" + ("" if bruteforceonly else " vs DFS"))

def addone(gamestate):
    i = 0
    while i<numstates and gamestate[i] == 4: i += 1
    if i == numstates: return None
    return [1 for _ in gamestate[:i]]+ [gamestate[i]+1] + gamestate[i+1:]

gamestate = [0 for _ in range(numstates)]


def verifyvic(gamestate):
    victory = True
    for start, nbs in adjset.items():
        for end in nbs:
            if end != start and start>=0 and end>=0:
                if gamestate[start]*gamestate[end]==0 or gamestate[start]==gamestate[end]:
                    victory = False
                    break
        if not victory: break
    return victory

done = False

if True:
    init = tuple(0 for _ in range(numstates))
    visited = {init}
    agenda = [init]
    while True:
        ax1.set_xticks([])
        ax1.set_yticks([])
        gamestate = addone(gamestate)
        if gamestate is None: break
        ax1.imshow(converter(gamestate), interpolation='nearest', cmap=cmap, norm =norm) 
        

        if not bruteforceonly:
            if agenda:
                last = agenda.pop()
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.imshow(converter(last), interpolation='nearest', cmap=cmap, norm =norm) 
                
                # search for first zero
                i = 0
                while i<numstates and last[i] != 0: i+=1
                if i == numstates: 
                    done=True
                    agenda = []
                else:
                    nbcolors = [last[j] for j in adjset[i] if last[j]!=0]
                    for k in [j for j in range(1,5) if j not in nbcolors]:
                        cur = last[:i] + (k,) + last[i+1:]
                        if cur not in visited:
                            visited.add(cur)
                            agenda.append(cur)
                
            else:
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.imshow(converter(last), interpolation='nearest', cmap=cmap, norm =norm) 
            
        plt.pause(0.01)
        ax1.cla()
        if not bruteforceonly: ax2.cla()

        if verifyvic(gamestate): 
            print(gamestate)
            break
  
# plt.savefig('pixel_plot.png') 
  
# show plot 
plt.show() 