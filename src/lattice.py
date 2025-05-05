import networkx as nx

class Lattice:
    def __init__(self, word_tokens: list[str], fine_grain_tokens: list[str], type: str):
        self.match = False
        self.iteration = False
        self.fgt_edge_pointer = 0 # Fine grain token edge pointer
        self.wt_edge_pointer = 0 # Word token edge pointer
        self.wt_node_pointer = 0 # Word token node pointer
        self.word_tokens = word_tokens
        self.fine_grain_tokens = fine_grain_tokens
        self.graph = self.__build_lattice(word_tokens, fine_grain_tokens, type)

    def __build_lattice(self, word_tokens, fine_grain_tokens, type: str):
        # Create a directed graph
        graph = nx.DiGraph()

        # Get the lengths of the word and fine-grain tokens
        word_tokens_length = len(word_tokens)
        fine_grain_tokens_length = len(fine_grain_tokens)

        # Looping through the word tokens - using them as an 'anchor' 
        # to the main nodes to build the lattice
        for _ in range(word_tokens_length):

            word_token = word_tokens[self.wt_edge_pointer]
            fine_grain_token = fine_grain_tokens[self.fgt_edge_pointer]
            try:
                # If the word and fine-grain tokens match, add an edge
                if word_token == fine_grain_token:
                    graph.add_edge(self.wt_node_pointer, self.wt_node_pointer + 1, token=word_token, type=("word",type))
                    # Increment the pointers
                    self.wt_edge_pointer += 1
                    self.fgt_edge_pointer += 1
                    self.wt_node_pointer += 1

                else:
                    """
                    If they don't match, still add an edge to the graph
                    But when they eventually match, we will add a new edge for the word token
                    """
                    fine_grain_start = self.fgt_edge_pointer
                    fine_grain_end = self.fgt_edge_pointer + 1

                    while True:
                        # This loop should not break on the first iteration as it checks the same condition from the previous if statement
                        sub_token_str = fine_grain_tokens[fine_grain_start:fine_grain_end]
                        sub_token_str = "".join(sub_token_str)

                        # Check if the fine grain token sub string matches the current word token
                        if word_token == sub_token_str:

                            # Add an edge to the graph for the whole word token
                            graph.add_edge(self.wt_node_pointer, fine_grain_end, token=word_token, type=("word"))

                            # Add an edge to the graph for the last fine-grain token that completes a match with the word token
                            graph.add_edge(fine_grain_end-1, fine_grain_end, token=fine_grain_tokens[fine_grain_end-1:fine_grain_end][0], type=(type))

                            # Set the pointers to the next relevant position (for edges or nodes)
                            self.fgt_edge_pointer = fine_grain_end
                            self.wt_edge_pointer += 1
                            self.wt_node_pointer = fine_grain_end
                            break

                        # If the fine-grain token sub string does not match the current word token
                        else:
                            graph.add_edge(fine_grain_end-1, fine_grain_end, token=fine_grain_tokens[fine_grain_end-1:fine_grain_end][0], type=(type))

                            # Increment the fine-grain token end pointer
                            fine_grain_end += 1

                            # If we reach the end of the fine-grain tokens, break the loop
                            if self.wt_edge_pointer >= fine_grain_tokens_length:
                                break

            except Exception:
                return None

        return graph


    def display_lattice(self):
        print("Lattice:")
        for start, end, data in self.graph.edges(data=True):
            print(f"Edge from {start} to {end}: {data['token']} ({data['type']})")


    def get_lattice_positional_encodings(self):
        # Store positional encodings for each edge in the lattice for each tokenisation
        word_lpes = []
        fine_grain_lpes = []

        all_lpes = {}

        # Create a dictionary to store positional encodings for each token
        for start, _, data in self.graph.edges(data=True):
            all_lpes[data["token"]] = start + 1

        # Form individual positional encodings for each tokenisation
        for token in self.word_tokens:
            word_lpes.append(all_lpes[token])

        for token in self.fine_grain_tokens:
            fine_grain_lpes.append(all_lpes[token])

        return word_lpes, fine_grain_lpes