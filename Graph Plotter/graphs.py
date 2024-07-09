import numpy as np 
import math, json, os
import matplotlib.pyplot as plt 
import matplotlib.animation as anime
import matplotlib.patches as mp
from matplotlib import colors
from matplotlib.widgets import Button

########### Parameters ###########
noderad = 0.6 # radius of the nodes, default 0.6
textsize = 15 # size of labels, default 15
margin = 2 # amount of white space from border, default 2
velocityscale = 0.01 # speed of flow dots, default 0.01
thickness = 1 # thickness of lines, default 1
nodebg = "white" # node color, default "white"
loadfilename = "" # name of saved graph, without .json
##################################

'''
Guide!!! :D

You will be able to save a graph after you plot it.
(recommend to frequently do so because matplotlib often crashes).
If you want to load a saved graph file (e.g. flownetwork.json),
put "flownetwork" into the loadfilename line above.

When the figure is run, a live terminal will be running in the background.
It will keep track the different inputs you type in or click on.
At any point of time, to erase the current input (e.g. after clicking 
the wrong keys), press Esc to cancel.

ADDING NODES:
- Click anywhere on the figure. Type a name of the node in LaTeX (e.g. x or \alpha) then Enter/Return.
** Note **
    (1) This will plot a node at the nearest lattice point.
    (2) Node name must be unique to the given coordinate.
    (3) Use the ` key for _, e.g. for d_2

ADDING EDGES:
- Click the first node, then the second node, then type a name for the edge (optional).
- If no other attributes are wanted, just press Enter/Return. Otherwise, see below
- Edges can have the following attributes:
    1. weight (w): weight assigned to the edge (will be displayed on edge too)
    2. bend (b): amount of curvature (recommended range: -1 (left) to 1 (right))
    3. flow (f): amount of flow along the edge (recommended range: 0 to 5)
    4. directedness: whether to be directed (dir) / undirected (und)
                     (default is dir, unless changed via process described shortly below)
- To assign attributes, use commas to separate each attribute, like this:
    >> b=0.5,f=1,und (this creates an edge with weight 0, bend 0.5, flow 1, and undirected)
    >> w=1,f=1 (this creates an edge with weight 1, bend 0, flow 1, and directed by default)
    >> (nothing inputted: this creates a directed edge with no label, flow, nor bend)
  then press Enter/Return. 
** Note **
    (1) Do not use spaces! 
    (2) The order of the attributes doesn't matter
    (3) To change the default directedness, type dir/und when the clickqueue is empty.

MOVING NODES:
- Click the node, click a new location, then Enter.

DELETING OBJECTS (nodes or edges):
- Click the object, then Backspace. (OR: type "del" then Enter/Return.)
** Note ** 
    (1) If you delete a node, all edges attached to it will also be deleted.

HIGHLIGHTING OBJECTS:
- Click the object, then type "hl" then Enter/Return.

TOGGLING FLOW:
- Click the Toggle Flow Button to toggle between showing/unshowing flow along edges.

SAVE GRAPH:
- Type a valid name (e.g. flownetwork), then click the Save Button.
** Note **
    (1) They will be saved into a folder called saved_graphs. 
    (2) This folder is created once you run this program for the first time!

RUNNING A BFS/DFS:
- Click the source node, then click BFS or DFS. The source node should be highlighted.
- Then keep clicking the Next button to show the next step in the algorithm.
- Once it's completed, clicking Next will remove all the highlights.
'''

####################
# CODE STARTS HERE #
####################

cur_dir = os.path.dirname(__file__) # directory of this file
relpath = lambda x: os.path.join(cur_dir, x)

# create folder if absent
if not os.path.exists(relpath('saved_graphs')):
    os.makedirs(relpath('saved_graphs'))

# activate LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "TeX",
    "font.monospace": 'Computer Modern Typewriter'
})

# disable all built-in keyboard matplotlib shortcuts, e.g. f being full-screen
for param, shortcuts in plt.rcParams.items():
    if type(shortcuts)==list:
        for let in "abcdefghijklmnopqrstuvwwxyz":
            if let in shortcuts:
                shortcuts.remove(let)

# create figure
fig, ax = plt.subplots()
fig.set(figheight=6, figwidth=15) # dimensions of window

nodeset = {} # maps node coordinates to node objects
edgeset = {} # maps edge coordinate pairs to edge objects
adjmat = {} # maps each node object n to a list [(nb,e), ...] where e is the edge from n to nb

class node:
    xrange = [-2, 2] # for diagram dimensions
    yrange = [-2, 2]

    def __init__(self,x,y,s):

        # very big scatter pt (circ)
        self.shape = ax.scatter([x],[y], s=noderad*1000, ec="black", color=nodebg, linewidth=thickness,zorder=1) 

        self.labelshape = ax.text(x,y,s=s,horizontalalignment='center',verticalalignment='center', size=textsize,zorder=3)
        
        self.mask = ax.text(x,y,s="",horizontalalignment='center',
                            verticalalignment='center', size=textsize, c="black", alpha=1, zorder = 2)

        self.coord = (x,y)

        self.label = s

        self.xrange[0] = min(self.xrange[0], x-margin)
        self.xrange[1] = max(self.xrange[1], x+margin)
        self.yrange[0] = min(self.yrange[0], y-margin)
        self.yrange[1] = max(self.yrange[1], y+margin)
        
        nodeset[(x,y)] = self
        adjmat[self] = []

    def highlight(self, boo, write=""):
        self.shape.set(ec="red" if boo else "black",
                       linewidth = 2 if boo else 1)
        self.mask.set(text=write if boo else "", 
                      x=self.coord[0],
                      y=self.coord[1] + 0.6 * noderad * (self.yrange[1]-self.yrange[0])/fig.get_figheight(),
                      c="red")

    def lowlight(self, boo, write="", color="black"):
        self.mask.set(text=write if boo else "", 
                      x=self.coord[0],
                      y=self.coord[1] + 0.6 * noderad * (self.yrange[1]-self.yrange[0])/fig.get_figheight(),
                      c=color)

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return self.label[1:-1]

    def __del__(self):
        self.shape.remove()
        self.labelshape.remove()
        self.mask.remove()

    def __eq__(self, other):
        return self.coord == other.coord
    
    def move(self, othercoord):
        del nodeset[self.coord]
        x2, y2 = othercoord
        nodeset[othercoord] = self

        self.coord = othercoord
        self.shape.set_offsets(np.array([[x2,y2]]))
        self.labelshape.set(x=x2, y=y2)

        self.xrange[0] = min(self.xrange[0], x2-margin)
        self.xrange[1] = max(self.xrange[1], x2+margin)
        self.yrange[0] = min(self.yrange[0], y2-margin)
        self.yrange[1] = max(self.yrange[1], y2+margin)


class edge:
    def __init__(self,node1,node2,weight="",bend = 0, arrow=True, flow=0):
        self.shape, self.labelshape, self.labelshape2 = arrow_(node1.coord, node2.coord, weight, bend, arrow)
        self.start = node1
        self.end = node2
        self.weight = weight
        self.bend = bend
        self.arrow = arrow
        self.flowvalue = flow
        self.flowdots = ax.scatter([],[],zorder=0.5,color="blue")
        edgeset.setdefault((node1.coord,node2.coord), [])
        edgeset[node1.coord,node2.coord].append(self)
        adjmat[self.start].append((self.end, self))
        if not arrow: 
            adjmat[self.end].append((self.start, self))

    def dispflow(self, pos):
        dist = 1/self.flowvalue if self.flowvalue!=0 else None
        if dist: 
            pts = points_on_line_(self.start.coord, self.end.coord, pos, dist, bend = self.bend)
            self.flowdots.set_offsets(np.array(pts))
        else:
            self.flowdots.set_visible(False)
    
    def highlight(self, boo):
        self.shape.set_color("red" if boo else "black")
        self.shape.set_linewidth(2 if boo else 1)

    def __hash__(self):
        return hash((self.start,self.end,self.bend))

    def __repr__(self):
        return self.start.__repr__() + "-" + self.end.__repr__()
    
    def __del__(self):
        self.shape.remove()
        self.flowdots.remove()
        self.labelshape.remove()
        if self.labelshape2 is not None: self.labelshape2.remove()


def bent_midpoint(coord1, coord2, bend):
    x1,y1 = coord1
    x2,y2 = coord2
    disp = np.array([[0,1],[-1,0]]) @ np.array([[x2-x1],[y2-y1]]) * bend * 0.5
    return list((disp+np.array([[(x1+x2)/2],[(y1+y2)/2]])).flatten())

def arrow_(p,q,weight,bend,arrow):
    '''draws arrow'''
    x1, y1 = p
    x2, y2 = q

    arrowstyle = mp.ArrowStyle("->",head_length=6, head_width=3) if arrow else mp.ArrowStyle("-")

    shape = mp.FancyArrowPatch((x1,y1),(x2,y2), 
                                arrowstyle=arrowstyle,
                                connectionstyle=mp.ConnectionStyle("arc3", rad=bend),
                                shrinkA=noderad*20, shrinkB=noderad*20, zorder=0, linewidth=thickness)
    ax.add_artist(shape)

    x3, y3 = bent_midpoint(p,q,bend)

    if weight != "": 
        labelshape2 = ax.scatter([x3],[y3], s=noderad*400, ec="none", color="white", linewidth=thickness,zorder=0) 
    else: labelshape2 = None

    labelshape = ax.text(x3,y3,s=str(weight),
                         horizontalalignment='center',verticalalignment='center', size=textsize)

    return shape, labelshape, labelshape2

def points_on_line_(p,q,pos,dist,bend=0):
    '''
    Given two coordinates p,q, return a list of coordinates on the segment pq
    that differ from adjacent ones by distance=dist, and such that some point
    is at position pos (0<pos<1).
    '''
    x1, y1 = p
    x2, y2 = q
    dist = dist/((x1-x2)**2 + (y1-y2)**2)**0.5
    linspace = [pos + k * dist for k in range(math.ceil(-pos/dist),math.floor((1-pos)/dist)+1)]
    cs = [complex(pt, 2 * pt * (pt-1) * bend) * complex(x2-x1, y2-y1) + complex(x1,y1)
          for pt in linspace]
    return [(round(c.real,3), round(c.imag,3)) for c in cs]

def reshape_diagram():
    '''
    Reshapes the diagram dimensions according to existing node coordinates.
    '''
    ax.set(xlim= node.xrange, ylim= node.yrange, aspect=1, xticks=[], yticks=[])


############################
# Button Related Functions #
############################


'''Toggle Flow'''

butt = Button(plt.axes([0, 0, 0.2, 0.05]), "\\textbf{Toggle Flow}", image=None, color='0.85', hovercolor='0.95')
showflow = False

def toggleflow(_):
    global showflow
    showflow = not showflow
butt.on_clicked(toggleflow)


'''Next'''

nextbutt = Button(plt.axes([0.9, 0, 0.1, 0.05]), "\\textbf{Next}", image=None, color='0.85', hovercolor='0.95')
mission = None

def nextstep(_):
    global clickqueue
    try: new = next(mission)
    except: 
        for _,es in edgeset.items():
            for e in es:
                e.highlight(False)
        for _,n in nodeset.items():
            n.highlight(False)
        return
    for obj in new:
        obj[0].highlight(True, *obj[1:])
    clickqueue = []
nextbutt.on_clicked(nextstep)


'''Save'''

savebutt = Button(plt.axes([0.2, 0, 0.1, 0.05]), "\\textbf{Save}", image=None, color='0.85', hovercolor='0.95')

def savegraph(_):
    global inputstatus
    for_json = {
        "nodes": [],
        "edges": []
    }
    for _,n in nodeset.items():
        for_json["nodes"].append((n.coord[0], n.coord[1], n.label))
    for _,es in edgeset.items():
        for e in es:
            for_json["edges"].append((e.start.coord, e.end.coord, e.weight, e.bend, e.arrow, e.flowvalue))
    
    with open("saved_graphs/" + inputstatus + ".json", "w") as f:
        json.dump(for_json,f)
    
    print("Saved as " + inputstatus + ".json")
    inputstatus = ""
    print('>>', inputstatus)

savebutt.on_clicked(savegraph)

def loadgraph(jsonname):
    with open(relpath("saved_graphs/" + jsonname + ".json"), "r") as f:
        file = json.load(f)
    for n in file["nodes"]:
        node(*n)
    for e in file["edges"]:
        p, q = tuple(e[0]), tuple(e[1])
        edge(nodeset[p], nodeset[q], *e[2:])
    reshape_diagram()


'''Decorator'''

def activatebutt(butt):

    def f(func):
        
        def activatealg(x):

            global clickqueue, mission

            if len(clickqueue)==0:
                print("Pick a source node first!")
                return

            mission = func(adjmat, nodeset[clickqueue[-1]]) 
            clickqueue = []
            nextstep(x)

        butt.on_clicked(activatealg)

        return func

    return f


'''BFS'''

bfsbutt = Button(plt.axes([0.8, 0, 0.1, 0.05]), "\\textbf{BFS}", image=None, color='0.85', hovercolor='0.95')

@activatebutt(bfsbutt)
def bfs(adj, source):
    visited = {source}
    levels = [{source}]
    cur_level = 0
    yield [(source,"\\textbf{"+str(cur_level)+"}")]
    while True:
        levels.append(set())
        if levels[cur_level] == set(): break
        for v in levels[cur_level]:
            for nb,e in adj[v]:
                if nb not in visited:
                    yield [(e,),(nb,"\\textbf{"+str(cur_level+1)+"}")]
                    visited.add(nb)
                    levels[cur_level+1].add(nb)
        cur_level += 1


'''DFS'''

dfsbutt = Button(plt.axes([0.7, 0, 0.1, 0.05]), "\\textbf{DFS}", image=None, color='0.85', hovercolor='0.95')

@activatebutt(dfsbutt)
def dfs(graph, source):
    visited = {source}
    stack = [(source,)]
    yield [(source,)]
    while stack:
        ne = stack[-1]
        if ne[0] not in visited:
            yield [(k,) for k in ne]
            visited.add(ne[0])
        remove_from_stack = True
        for next_node, par_edge in graph[ne[0]]:
            if next_node not in visited:
                stack.append((next_node,par_edge))
                remove_from_stack = False
                break
        if remove_from_stack:
            stack.pop()


'''Dijkstra'''

dijksbutt = Button(plt.axes([0.55, 0, 0.15, 0.05]), "\\textbf{Dijkstra}", image=None, color='0.85', hovercolor='0.95')

@activatebutt(dijksbutt)
def dijkstra(graph, source):

    def extractmin(dictionary):
        return min(list(dictionary.items()), key=lambda x:x[1])[0]
    
    fakedist = {node: float("inf") for node in graph}
    fakedist[source] = 0

    for n in fakedist:
        n.lowlight(True, "$\\infty$" if fakedist[n] == float("inf") else str(fakedist[n]), "blue")

    truedist = {}
    
    while fakedist:
        popped = extractmin(fakedist)
        poppeddist = fakedist[popped]
        
        truedist[popped] = poppeddist

        for nb, e in graph[popped]:
            if nb in fakedist:
                fakedist[nb] = min(fakedist[nb], poppeddist + (e.weight if e.weight!="" else 0))
                nb.lowlight(True, 
                            "$\\infty$" if fakedist[nb] == float("inf") else str(fakedist[nb]), 
                            "blue")

        foundedge = False
        for other in truedist:
            for nb, e in graph[other]:
                if nb == popped and poppeddist == truedist[other] + e.weight:
                    foundedge = True
                    break
            if foundedge: break

        nodeyield = [(popped,"\\textbf{"+str(poppeddist)+"}")]

        yield nodeyield + [(e,)] if foundedge else nodeyield

        del fakedist[popped]


##########################
# Input and Click System #
##########################


inputstatus = '' # Input space

clickqueue = [] # Sequence of clicked coordinates

defaultarrow = True # plot directed edges by default?

def process_input():
    global clickqueue, nodeset, inputstatus, nodeset, edgeset, adjmat, defaultarrow

    if len(clickqueue) == 0: # Click Queue: 
        if inputstatus == "und":
            defaultarrow = False
            print("Default: Undirected edges")
        if inputstatus == "dir":
            defaultarrow = True
            print("Default: Directed edges")

    if len(clickqueue) == 1: # Click Queue: coord0
        coord0 = clickqueue[0]
        if coord0 not in nodeset:
            node(*coord0,'$'+inputstatus+'$' if inputstatus else '')
            print("Created node")
        if coord0 in nodeset:
            if inputstatus == "del":
                del adjmat[nodeset[coord0]]

                for n in adjmat:
                    adjmat[n] = [ve for ve in adjmat[n] if ve[0].coord != coord0]

                edgeset = {ee: e for ee,e in edgeset.items() if ee[0] != coord0 and ee[1] != coord0}

                del nodeset[coord0]
                print("Deleted node")

            elif inputstatus == "hl":
                nodeset[coord0].highlight(True)
                print("Highlighted node")

    if len(clickqueue) == 2 and clickqueue[0] in nodeset and clickqueue[1] in nodeset: # Click Queue: coord0 coord1
        
        coord0, coord1 = clickqueue

        if inputstatus == "hl":
            for e in edgeset[(coord0,coord1)]:
                e.highlight(True)
            print("Highlighted edge(s)")

        elif inputstatus == "del":
            
            for _, nb in adjmat.items():
                for ve in nb:
                    if ve[1].start.coord == coord0 and ve[1].end.coord == coord1:
                        nb.remove(ve)

            for e in edgeset[(coord0,coord1)]:
                del e

            del edgeset[(coord0,coord1)]
            print("Deleted edge(s)")

        else:
            p,q = nodeset[coord0], nodeset[coord1]
            splitinput = inputstatus.split(",")

            fv, bn, ar, we = 0, 0, defaultarrow, ""
            for prop in splitinput:
                if "f=" in prop:
                    fv = float(prop[2:])
                if "b=" in prop:
                    bn = float(prop[2:])
                if prop=="und":
                    ar = False
                if prop=="dir":
                    ar = True
                if "w=" in prop:
                    try:
                        we = int(prop[2:])
                    except:
                        we = float(prop[2:])

            edge(p,q, weight = we, flow = fv, bend = bn, arrow=ar)

            print("Created edge")

    if len(clickqueue) == 2 and clickqueue[0] in nodeset and clickqueue[1] not in nodeset:

        coord0, coord1 = clickqueue

        nodeset[coord0].move(coord1)
        
        for edgecoord in edgeset.copy():
            if edgecoord[0] == coord0:
                for e in edgeset[edgecoord]:
                    e.shape.set_positions(coord1, edgecoord[1])
                    xm, ym = bent_midpoint(coord1, edgecoord[1], e.bend)
                    e.labelshape.set(x=xm, y=ym)
                    if e.labelshape2 is not None:
                        e.labelshape2.set_offsets(np.array([[xm,ym]]))
                edgeset[(coord1, edgecoord[1])] = edgeset[edgecoord]
                del edgeset[edgecoord]
                
            if edgecoord[1] == coord0:
                for e in edgeset[edgecoord]:
                    e.shape.set_positions(edgecoord[0], coord1)
                    xm, ym = bent_midpoint(edgecoord[0], coord1, e.bend)
                    e.labelshape.set(x=xm, y=ym)
                    if e.labelshape2 is not None:
                        e.labelshape2.set_offsets(np.array([[xm,ym]]))
                edgeset[(edgecoord[0], coord1)] = edgeset[edgecoord]
                del edgeset[edgecoord]

        print("Moved node")

    clickqueue = []


def onclick(event):
    global clickqueue
    x,y = round(event.xdata), round(event.ydata)
    if event.inaxes.get_position().bounds[1] != 0:
        clickqueue.append((x,y))
        print('Click Queue:', *clickqueue)
fig.canvas.mpl_connect('button_press_event', onclick)


def onkey(event):
    global inputstatus, clickqueue, nodeset, mission
    
    entered = event.key
    
    toprint = True

    if entered == 'escape':
        inputstatus = ''
        clickqueue = []
        mission = None
        nextstep(0)
    elif entered == 'enter':
        process_input()
        inputstatus = ""
    elif entered == 'backspace':
        inputstatus = "del"
        process_input()
        inputstatus = ""
    elif len(entered)==1:
        inputstatus += '_' if entered=='`' else entered
    else:
        toprint = False
    
    if toprint: print('>>', inputstatus)
    reshape_diagram()
fig.canvas.mpl_connect('key_press_event', onkey)


def update(frame):
    if showflow:
        for _,edges in edgeset.items():
            for edge in edges:
                edge.flowdots.set_visible(True)
                edge.dispflow(frame*velocityscale)
    else:
        for _,edges in edgeset.items():
            for edge in edges:
                edge.flowdots.set_visible(False)

ani = anime.FuncAnimation(fig=fig, func=update, frames=1000, interval=30)

if loadfilename: loadgraph(loadfilename)


################
# Manual Build #
################

'''Note: Objects built here may not be removeable (due to additional pointers), 
but you can save it and then reload it as a new graph, then they are removeable.'''

# N,R = 15,10

# margin = 3

# for i in range(N):
#     node(-int(R*math.sin(2*math.pi*i/N)),int(R*math.cos(2*math.pi*i/N)), str(i+1))

# for i in range(N):
#     for j in range(i+1,N):
#         edge(tenrings[i],tenrings[j],arrow=False)

################
################
################

reshape_diagram()

plt.show() 
fig.show()
# ani.save(filename="graph.gif", writer="pillow")