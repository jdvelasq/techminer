class MainPath:
    def __init__(self, nodes):

        result = {}
        for key in nodes.keys():
            for target in nodes[key]:
                if target in result.keys():
                    result[target] += [key]
                else:
                    result[target] = [key]
        self.nodes = result

    def search_sources(self):
        sources = set(self.nodes.keys())
        targets = set(target for key in self.nodes.keys() for target in self.nodes[key])
        self.sources = list(sources - targets)

    def build_paths(self):

        stack = [[x] for x in self.sources]
        paths = []

        while True:

            current_path = stack.pop(0)
            last_element = current_path[-1]

            if last_element in self.nodes.keys():
                next_nodes = self.nodes[last_element]
                for next_node in next_nodes:
                    stack.append(current_path.copy() + [next_node])
            else:
                paths.append(current_path.copy())

            if len(stack) == 0:
                break

        self.paths = paths

    def search_path_count(self):

        links = {}
        for path in self.paths:
            for i, _ in enumerate(path[:-1]):
                key = (path[i], path[i + 1])
                if key in links.keys():
                    links[key] += 1
                else:
                    links[key] = 1

        self.links = links

    def search_significant_links(self):
        max_value = max(self.links.values())
        return [key for key in self.links.keys() if self.links[key] == max_value]

    def global_key_route_search(self):

        most_significant_paths = []

        #
        # Computes sum of links for each path
        #
        path_values = {}
        for path in self.paths:
            path_values[tuple(path)] = sum(
                [self.links[(path[i], path[i + 1])] for i, _ in enumerate(path[:-1])]
            )

        #
        # Paths with most significant links
        #
        significant_links = self.search_significant_links()
        for a, b in significant_links:
            paths = [path for path in self.paths if a in path and b in path]
            max_value = max([path_values[tuple(path)] for path in paths])
            selected_paths = [
                path for path in paths if path_values[tuple(path)] == max_value
            ]
            most_significant_paths.extend(selected_paths)

        #
        # Most significant paths
        #
        max_value = max(path_values.values())
        selected_paths = [
            list(key) for key in path_values.keys() if path_values[key] == max_value
        ]
        most_significant_paths.extend(selected_paths)

        #
        # Drop duplicates
        #
        most_significant_paths = [tuple(path) for path in most_significant_paths]
        most_significant_paths = list(set(most_significant_paths))
        self.most_significant_paths = [list(path) for path in most_significant_paths]

        self.global_key_route_paths = sorted(
            set(
                (path[i], path[i + 1])
                for path in self.most_significant_paths
                for i, _ in enumerate(path[:-1])
            )
        )
