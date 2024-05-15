from collections import deque

class Edge:
    def __init__(self, u, v, nxt, cap):
        self.u = u
        self.v = v
        self.nxt = nxt
        self.cap = cap
        self.flow = 0
    
class Graph:
    def __init__(self, n_nodes):
        self.e = []
        self.fst = [-1] * n_nodes
    
    def add(self, u, v, cap1, cap2):
        self.e.append(Edge(u, v, self.fst[u], cap1))
        self.fst[u] = len(self.e) - 1
        self.e.append(Edge(v, u, self.fst[v], cap2))
        self.fst[v] = len(self.e) - 1

    def mark_affiliation(self, _s):
        self.aff = [1] * len(self.fst)
        q = deque()
        self.aff[_s] = 0
        q.append(_s)
        while q:
            cur = q.popleft()
            i = self.fst[cur]
            while i != -1:
                e = self.e[i]
                if e.cap > e.flow and self.aff[e.v] == 1:
                    self.aff[e.v] = 0
                    q.append(e.v)
                i = e.nxt

    # Edmonds-Karp algorithm for max flow
    def maxflow(self, _s, _t):
        flow = 0
        ind = 0
        while True:
            q = deque()
            q.append(_s)
            self.p = [-1] * len(self.fst)
            self.p[_s] = -2
            while q and self.p[_t] == -1:
                cur = q.popleft()
                i = self.fst[cur]
                while i != -1:
                    e = self.e[i]
                    if e.cap > e.flow and self.p[e.v] == -1:
                        self.p[e.v] = i
                        q.append(e.v)
                    i = e.nxt
            if self.p[_t] == -1:
                break
            push = 1 << 30
            cur = _t
            while cur != _s:
                e = self.e[self.p[cur]]
                push = min(push, e.cap - e.flow)
                cur = e.u
            cur = _t
            while cur != _s:
                e = self.e[self.p[cur]]
                e.flow += push
                self.e[self.p[cur] ^ 1].flow -= push
                cur = e.u
            ind += 1
            # the following line is to prove it's runnning but not dead
            # you can comment it for better performance
            print("update the {}th time, push = {}, flow = {}".format(ind, push, flow))
            flow += push
        return flow