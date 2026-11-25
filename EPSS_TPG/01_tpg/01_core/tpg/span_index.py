"""Insertion-order-preserving interval queries for immutable extraction spans."""
from bisect import bisect_left, bisect_right


class SpanIndex:
    def __init__(self, items, span):
        self.items = list(items)
        self.rows = sorted(((*span(item), i) for i, item in enumerate(self.items)))
        self.starts = [r[0] for r in self.rows]
        self.size = 1
        while self.size < len(self.rows):
            self.size *= 2
        self.max_end = [float('-inf')] * (2 * self.size)
        for i, (_, end, _) in enumerate(self.rows):
            self.max_end[self.size + i] = end
        for i in range(self.size - 1, 0, -1):
            self.max_end[i] = max(self.max_end[2*i], self.max_end[2*i+1])

    def _query(self, hi, end, strict):
        found, stack = [], [(1, 0, self.size)]
        while stack:
            node, lo, right = stack.pop()
            if lo >= hi or (self.max_end[node] <= end if strict else self.max_end[node] < end):
                continue
            if right - lo == 1:
                found.append(self.rows[lo][2])
            else:
                mid = (lo + right) // 2
                stack.extend(((node*2+1, mid, right), (node*2, lo, mid)))
        return [self.items[i] for i in sorted(found)]

    def containing(self, start, end):
        return self._query(bisect_right(self.starts, start), end, False)

    def at(self, point):
        return self._query(bisect_right(self.starts, point), point, True)

    def overlapping(self, start, end):
        return self._query(bisect_left(self.starts, end), start, True)


def node_span(node):
    return node.properties.char_start, node.properties.char_end


def graph_index(graph, kind):
    # Frontends assign source offsets at node creation; the extraction passes do
    # not mutate them. New nodes invalidate only their own node-type index.
    cache = getattr(graph, '_span_indexes', None)
    if cache is None:
        graph._span_indexes = cache = {}
    size = len(graph._nodes_by_type.get(kind, ()))
    previous = cache.get(kind)
    if previous is None or previous[0] != size:
        previous = cache[kind] = (size, SpanIndex(graph.nodes(kind), node_span))
    return previous[1]
