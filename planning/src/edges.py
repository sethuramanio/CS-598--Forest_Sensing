import numpy as np
from graph import Graph
from isvalids import Valids
import utils

class GraphEdgeMaker:
    def __init__(self, img, visibilityWindow, vc=None):
        self.img = img
        self.visibilityWindow = visibilityWindow
        self.vc = vc

    def fillGrid(self, samples, g):
        for s in samples:
            g.addVertex(s)

        def add_edge_to_graph(p1, p2):
            points = self.discretizeEdge(p1, p2)
            valid = True
            for point in points:
                if not self.vc(point, []):
                    valid=False
            if valid:
                g.addEdge(p1, p2, utils.euclidean(p1, p2))
                return True
            return False

        np_samples = np.array(samples)
        x_coords = np.unique(np_samples[:, 0])
        x_coords.sort()
        y_coords = np.unique(np_samples[:, 1])
        y_coords.sort()

        missed_points = []

        for ex in range(x_coords.shape[0]):
            for why in range(y_coords.shape[0]):
                x = x_coords[ex]
                y = y_coords[why]
                coord = (x, y)

                coords_to_add = []
                if ex > 0:
                    prev_x = x_coords[ex-1]
                    coords_to_add.append((prev_x, y))
                if ex < len(x_coords)-1:
                    coords_to_add.append((x_coords[ex+1], y))
                if why > 0:
                    coords_to_add.append((x, y_coords[why-1]))
                if why < len(y_coords)-1:
                    coords_to_add.append((x, y_coords[why+1]))

                for c in coords_to_add:
                    if not add_edge_to_graph(coord, c):
                        missed_points.append(coord)
                        missed_points.append(c)

        mpt = np.empty(len(missed_points), dtype=object)
        mpt[:] = missed_points
        mpt = np.unique(mpt)
        for i in range(len(mpt)):
            for j in range(i):
                p1 = mpt[i]
                p2 = mpt[j]
                if utils.euclidean(p1, p2) < 3*self.visibilityWindow:
                    add_edge_to_graph(p1, p2)
                
        return g

    def lawnmower(self, samples, g):
        g = self.fillGrid(samples, g)
        return self.lawnmowerTraverse(samples, g)

    def lawnmowerTraverse(self, samples, g):

        ccs = g.getCCs()

        seen_ccs = [0] # ccs we can't visit again.
        path = []
        length = np.inf
        next_start_point = ccs[seen_ccs[-1]][0]
        for i in range(len(ccs)):
            p, l = self.gridTSP(g, next_start_point, max_length=len(ccs[seen_ccs[-1]]))
            last_point = p[-1]

            # Check the other CCS, pick the cc to go to next. 
            best_cc_number = -1
            best_length = np.inf
            for next_cc in range(len(ccs)):
                if next_cc in seen_ccs:
                    continue
                edges, lengths = self.makeBridge([last_point], ccs[next_cc])
                if lengths[0] < best_length:
                    best_length = lengths[0]
                    best_cc_number = next_cc
                    next_start_point = edges[0][1]

            seen_ccs.append(best_cc_number)
            length += best_length

            path = path + list(p)
            length += l


        return g, path, length

    def localNetworkAlgorithm(self, samples, g, **args):
        # build components. 
        g = self.drawValidOnly(samples, g)

        # Make bridges
        ccs = g.getCCs()
        if len(ccs) == 1:
            p, l =  self.bruteForceShortestHamiltonian(g, samples[0])
            return g, p, l

        bridge_edges, bridge_ccs, bridge_lengths = self.makeAllBridges(samples, g)

        # Construct a bridgeGraph for shortest crossover paths. 
        bridgeGraph = Graph()
        for i in np.unique(bridge_ccs):
            bridgeGraph.addVertex(i)

        for i in range(len(bridge_ccs)):
            edge = bridge_ccs[i]
            length = bridge_lengths[i]
            bridgeGraph.addEdge(edge[0], edge[1], length)
     
        # figure out order of ccs
        bridge_graph, bridge_path, over_water_length = self.nearestNeighborTSP(bridgeGraph.getVertices(), bridgeGraph, useGraphEdges=True)
        # Get order and endpoints. 
        # [0, 1, 2] -> [(0, 1), (1, 2)] -> endpoints
        endpoints = []
        for i in range(len(bridge_path)-1):
            try:
                bridge_edge = (bridge_path[i], bridge_path[i+1])
                bridge_idx = bridge_ccs.index(bridge_edge)
                start, end = bridge_edges[bridge_idx]
            except:                
                bridge_edge = (bridge_path[i+1], bridge_path[i])
                bridge_idx = bridge_ccs.index(bridge_edge)
                end, start = bridge_edges[bridge_idx]
                
            endpoints.append(start)
            endpoints.append(end)

        # Find the paths and put it all together
        # We have endpoints 
        # [0, 1, 2, 3, ..., n]
        # = [(start anywhere), 0], [1, 2], ..., [n, (end anywhere)]
        endpoints = [None] + endpoints + [None]
        path = []
        total_length = 0
        for i in range(int(len(endpoints)/2)):
            # cc_number = bridge_path[i]
            start_idx = 2*i
            pt1 = endpoints[start_idx]
            pt2 = endpoints[start_idx+1]

            subpath = []
            max_cc_length = len(ccs[i])
            if pt1 is None:
                p, length = self.bruteForceShortestHamiltonian(g, start=pt2, max_length=max_cc_length)
                subpath = p[::-1]
            else:
                subpath, length = self.bruteForceShortestHamiltonian(g, pt1, pt2, max_length=max_cc_length)

            path += subpath
            total_length += length

        # The above calculations include over water lengths
        total_length += over_water_length

        return g, path, length

    def bruteForceShortestHamiltonian(self, g, start, end=None, max_length=-1):
        assert(start != None)
        if end is not None and start == end and max_length == 1:
            return [start], 0
        if max_length == -1:
            max_length = g.nVertices()
        if start == end:
            max_length += 1
        
        experience = {1:[]} # path length to list of path numbers
        paths = {} # path number to ([path], length). 
        npaths = 0

        # subset_length = 1
        paths[npaths] = ([start], 0)
        experience[1].append(npaths)
        npaths += 1

        # get our neighbors. 
        leaves = [(0, n) for n in g.getNeighbors(start)] #(pathid, nextElement)

        # commence brute force. 
        for subset_length in range(2, max_length+1):
            if len(leaves) == 0:
                break
            experience[subset_length] = []
            new_leaves = []
            for leaf in leaves:
                pathid, j = leaf
                prev_path, prev_length = paths[pathid]

                new_path = prev_path + [j]
                new_leg_length = g.getEdge(prev_path[-1], j)

                new_path_id = npaths
                paths[new_path_id] = (new_path, prev_length + new_leg_length)
                experience[subset_length].append(new_path_id)

                if j == end and len(prev_path) > 2: # we're seeing end and it's not the start.  
                    continue 

                for neighbor in g.getNeighbors(j):
                    if neighbor not in prev_path:
                        new_leaves.append((new_path_id, neighbor))                
                    if end is not None and neighbor == end:
                        new_leaves.append((new_path_id, neighbor))
            
                npaths += 1

            leaves = new_leaves.copy()

        # the paths are ripe for the picking
        fullTraversalLength = min(np.max(list(experience.keys())), max_length)
        shortlisted_paths = []
        path_indices = experience[fullTraversalLength]

        if end is None: # get any ol shortest path
            shortlisted_paths = path_indices
        else: # get the shortest path with desired end
            for path_index in path_indices:
                path, length = paths[path_index]
                if end == path[-1]:
                    shortlisted_paths.append(path_index)

        
        best_path = []
        best_length = np.inf
        for path_index in shortlisted_paths:
            path, length = paths[path_index]
            if length < best_length:
                best_length = length
                best_path = path

        return best_path, best_length

    def kNearestNeighbors(self, vertices, v, k=5):
        if len(vertices) == 0:
            return v, [0]
        if len(vertices) < k:
            k = len(vertices)

        samples_x_coords = np.array(vertices)[:,0]
        samples_y_coords = np.array(vertices)[:,1]
        s_x_coords = np.ones(len(vertices)) * v[0]
        s_y_coords = np.ones(len(vertices)) * v[1]
        euclideans = np.sqrt(np.square(samples_x_coords-s_x_coords)+np.square(samples_y_coords-s_y_coords))
        
        euclideans[np.where(euclideans == 0)] = np.inf # don't pick current element as its nearest neighbor
        min_indices = euclideans.argsort()[:k]

        verts = np.empty(len(vertices), dtype=object)
        verts[:] = vertices
        return verts[min_indices], euclideans[min_indices]

    def drawNaive(self, samples, graph, k=5):
        for sample in samples:
            nearest, dists = self.kNearestNeighbors(samples, sample, k)
            for n, d in zip(nearest, dists):
                graph.addEdge(sample, n, d)
        return graph

    def drawValidOnly(self, samples, graph, k=5):
        v = self.vc# Valids(self.img, self.visibilityWindow)
        for sample in samples:
            nearest, dists = self.kNearestNeighbors(samples, sample, k)
            for n, d in zip(nearest, dists):
                points = self.discretizeEdge(sample, n)
                valid = True
                for point in points:
                    if not self.vc(point, []):
                        valid = False
                if valid:
                    if d != np.inf:
                        graph.addEdge(sample, n, d)

        return graph

    def discretizeEdge(self, v1, v2, n=10):
        x_coords = np.linspace(v1[0], v2[0], n)
        y_coords = np.linspace(v1[1], v2[1], n)
        return np.vstack((x_coords, y_coords)).astype(np.int).T

    def makeAllBridges(self, samples, g):
        components = g.getCCs()
        lengths = []
        edges = []
        components_connecting = []
        for i in range(len(components)):
            for j in range(i):
                # Connect components i and j
                bridge_edges, bridge_lengths = self.makeBridge(components[i], components[j])
                # g.addEdge(edge[0], edge[1], length)
                for e, l in zip(bridge_edges, bridge_lengths):
                    edges.append(e)
                    components_connecting.append((i, j))
                    lengths.append(l)

        # return g
        # return [edges], [(components connecting)], [lengths]
        return edges, components_connecting, lengths
    
    # connects two components with k bridges.
    def makeBridge(self, c1, c2, k=2):
        assert(type(k) is int)
        starter_list = c1.copy()
        best_edges = np.empty(k, dtype=object)
        best_lengths = np.ones(k) * np.inf
        for point in starter_list:
            nearest, length = self.kNearestNeighbors(c2, point, k)
            for n, l in zip(nearest, length):
                max_index = np.argmax(best_lengths)
                if l < best_lengths[max_index]:
                   best_edges[max_index] = (point, n)
                   best_lengths[max_index] = l
        return best_edges, best_lengths

    def smartTSP(self, samples, g, **args):
        k = max(4, int(len(samples) / 5))
        g = self.drawNaive(samples, g, k=k)
        g, path, length = self.nearestNeighborTSP(samples, g, useGraphEdges=True)
        return g, path, length

    # lot of copy paste here. Finds points on a grid. does some backtracking when cornered
    def gridTSP(self, g, start_point, max_length=-1):
        if max_length == -1:
            max_length = g.nVertices

        adj = g.getGraph().copy()
        all_verts = g.getVertices().copy()
        original_start_point = start_point
        tuples = type(all_verts[0]) is tuple

        # our path is going to be a list of indices of g.getVertices()
        # (until later converted)
        all_verts = list(range(g.nVertices()))
        start_point = g.getVertices().index(start_point)
        
        # all_verts is guaranteed to be a list.

        path = [start_point]    
        skipped = []
        length = 0
        longest_path = [start_point]
        while len(path) + len(skipped) != max_length:#g.nVertices():
            last_point = path[-1]
            index = all_verts.index(last_point)

            # all of our neighbors
            neighbors = adj[:, index].copy()
            neighbors = np.trunc(neighbors / 10.) * 10
            neighbors[np.where(neighbors == 0)] = np.inf # don't pick nonedges
            # Find who we've seen
            neighbors[path] = np.inf # people we've seen are unvisitable again
            neighbors[skipped] = np.inf


            # our candidates
            min_indices = np.where(neighbors == np.min(neighbors))[0]


            if len( min_indices) == 0:
                break
        
            if neighbors[min_indices[0]] == np.inf:
                skipped.append(last_point)
                path = path[:-1]
                if len(path) == 0:
                    path = longest_path
                    break
                continue

            chosen_index = np.random.choice(min_indices)

            if len(min_indices) > 1 and tuples:
                other_coords = g.numpy_verts[min_indices] # The other coordinate options.
                my_coord = g.numpy_verts[last_point]
                coords_not_tuples = np.array(g.getVertices())[min_indices] 
                # The other coodinate options, as a np 
      
                # prioritize up, then down. 
                # left, then right. 
                coords_above = other_coords[np.where(coords_not_tuples[:, 1] >= my_coord[1])]
                coords_below = other_coords[np.where(coords_not_tuples[:, 1] <= my_coord[1])]
                coords_right = other_coords[np.where(coords_not_tuples[:, 0] <= my_coord[0])]
                coords_left = other_coords[np.where(coords_not_tuples[:, 0] >= my_coord[0])]

                top_right = np.intersect1d(coords_above, coords_right)
                top_left = np.intersect1d(coords_above, coords_left)
                bottom_right = np.intersect1d(coords_below, coords_right)
                bottom_left = np.intersect1d(coords_below, coords_left)

                if False:
                    continue
                elif len(top_right) > 0:
                    chosen_coord = top_right[-1]
                elif len(bottom_right) > 0:
                    chosen_coord = bottom_right[-1]
                elif len(top_left) > 0:
                    chosen_coord = top_left[-1]
                elif len(bottom_left) > 0:
                    chosen_coord = bottom_left[-1]

                chosen_index = g.getVertices().index(chosen_coord)
            
            # pick the guy if we have his vertex
            if neighbors[chosen_index] != np.inf:
                next_element = all_verts[chosen_index]
                path.append(next_element)
                length += neighbors[chosen_index]
            else: # skip this guy, just don't add him to the path. 
                skipped.append(all_verts[chosen_index])

            if len(path) > len(longest_path):
                longest_path = path

            assert(length != np.inf)

        if tuples:
            path = g.numpy_verts[path]

        assert(len(path) > 0)
        return path, length

    # only uses the edges that are already in the graph. 
    def nearestNeighborsTSP_limited(self, g, start_point):
        return self.gridTSP(g, start_point)
        adj = g.getGraph().copy()
        all_verts = g.getVertices().copy()
        original_start_point = start_point
        tuples = type(all_verts[0]) is tuple

        if tuples:
            all_verts = list(range(g.nVertices()))
            start_point = g.getVertices().index(start_point)
        
        # verts.remove(start_point)

        path = [start_point]
        skipped = []
        length = 0
        while len(path) != g.nVertices():
            last_point = path[-1]
            index = all_verts.index(last_point)

            # all of our neighbors
            neighbors = adj[:, index]
            neighbors[np.where(neighbors == 0)] = np.inf # don't pick current element as its nearest neighbor
            min_indices = neighbors.argsort()

            # remove the neighbors who we've already seen
            seen_indices =  np.argmax(min_indices == np.array(path+skipped)[:, np.newaxis], axis=1)
            min_indices = np.delete(min_indices, seen_indices)

            if len(min_indices) == 0:
                break

            # pick the guy if we have his vertex
            if neighbors[min_indices[0]] != np.inf:
                next_element = all_verts[min_indices[0]]
                path.append(next_element)
                length += neighbors[min_indices[0]]
            else: # skip this guy, just don't add him to the path. 
                skipped.append(all_verts[min_indices[0]])

            # assert(length != np.inf)

        if tuples:
            path = g.numpy_verts[path]

        return path, length

    def nearestNeighborTSP(self, samples, g, **args):# useGraphEdges=False):
        # deal with args
        if "useGraphEdges" not in args:
            useGraphEdges = False
        else:
            useGraphEdges = args["useGraphEdges"]
        
        # our four best possible start vertices are rightmost, leftmost, topmost, and bottommost. 
        # probably. 
        # but only do this if our samples are tuples.
        if type(samples[0]) is tuple:
            s_temp = np.array(samples) # this is NOT an array of tuples
            top_and_right = np.argmax(s_temp, axis=0)
            bottom_and_left = np.argmin(s_temp, axis=0)
            start_verts = [samples[i] for i in top_and_right] + \
                            [samples[j] for j in bottom_and_left]
            bestPath = []
            bestLength = np.inf
        else: # 10 random start points. 
            start_verts = np.random.choice(samples, size=10)

        # some heuristics aren't worth it.
        if len(samples) < 10:
            start_verts = samples

        bestPath = []
        bestLength = np.inf
        for start_vert in start_verts:
            if not useGraphEdges:
                path, length = self.nearestNeighborTSP_one(samples, start_vert)
            if useGraphEdges:
                path, length = self.nearestNeighborsTSP_limited(g, start_vert)

            if length < bestLength:
                bestLength = length
                bestPath = path

        # Add to graph g.
        if not useGraphEdges:
            for i in range(len(bestPath)-1):
                g.addEdge(bestPath[i], bestPath[i+1])

        return g, bestPath, bestLength

    def nearestNeighborTSP_one(self, samples, start_vertex):
        path = [start_vertex]
        pathLength = 0
        samples_copy = samples.copy()
        samples_copy.pop(samples_copy.index(start_vertex))
        while len(path) != len(samples):
            nearest, length = self.kNearestNeighbors(samples_copy, path[-1], 1)
            nearest = nearest[0]
            length = length[0]
            
            path.append(nearest)
            pathLength += length
            samples_copy.pop(samples_copy.index(nearest))
                
        return path, pathLength