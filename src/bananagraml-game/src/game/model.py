"""
- this is the data model that we adjust and use to 
  verify the game's current state.
"""
import math
import random
import uuid

dictionary = None
with open("dictionary.txt", encoding="utf-8") as f:
    dictionary = set(line.strip().upper() for line in f)


class BananaGramlModel:
    def __init__(self, BOARD_DIMENSIONS: (int, int, int)):
        self.board_valid = True
        self.coordinates = self.build_coordinates(coordinates=BOARD_DIMENSIONS)
        self.coordinate_ref = self.build_coordinate_ref()
        self.board = []
        self.tile_bank = TileBank()
        self.tiles_on_board = []  # need to make this the live board rep.
        self.tiles_on_bench = []

    def board_tiles(self):
        return self.tiles_on_board

    # TODO finalize this.
    def validate(self):
        # clean up the board and re-build it during each validate() call.
        self.clean_board()
        for tile in self.tiles_on_board:
            center = tile.model_tile.get_position()
            if center in self.coordinate_ref:
                x, y = self.coordinate_ref[center]
                self.board[x][y] = tile.model_tile

        """
        FIXME finish this at some point.
        isnot_disconnected_board = self.validate_disconnected_tiles(self.board)
        print(isnot_disconnected_board)
        if isnot_disconnected_board:
            return False
        """

        is_valid = self.build_words("", 0, 0, self.board)
        return is_valid

    def validate_words(self, words: [str]) -> bool:
        for word in words:
            if not (self.check_dictionary(word.upper())):
                return False
        return True

    def check_dictionary(self, word: str) -> bool:
        if word in dictionary:
            return True
        return False

    def validate_disconnected_tiles(self, board=None):
        """
        check if each tile is connected
        we are checking if we have a disconnected tile here.
        """
        if board is None:
            return False
        for i in range(0, len(board)):
            for j in range(0, len(board[i])):
                if board[i][j] == None:
                    continue
                if board[i][j] is not None:
                    if (
                        board[i - 1][j] is None
                        and board[i + 1][j] is None
                        and board[i][j - 1] is None
                        and board[i][j + 1] is None
                    ):
                        return False
        return True

    def build_words(self, word: str, row: int, column: int, board=None):
        """
        we can do a backtracking thing here.
        :: we're building a list of strings here
            -> if we see a tile, we keep reading right until we hit a NULL
            -> if we see a tile, we keep reading down until we hit a NULL
        :: edge cases
            -> we *don't* want to read subsets of words, ie
                word is EACH
                start at E
                read all the way to H
                start at A
                read all the way to H
                etc, etc.
                this would give us four words, with three of them being invalid.
        :: we want to consider this:
            -> what is the base case, where we finish the recursion.
                maybe instead of building a list of words, we can just check if the word is valid when
                we hit our base case, which in this case is
                    1. the cell on the right of a tile is NULL
                    2. the cell below the tile is NULL
                we would be building out the string as we iterate.
        """

        def check_row(i, j, board):
            word = ""
            while board[i][j] is not None:
                word += board[i][j].get_value()
                j += 1
            return word
            # return self.validate_words([word])

        def check_col(i, j, board):
            word = ""
            while board[i][j] is not None:
                word += board[i][j].get_value()
                i += 1
            return word
            # return self.validate_words([word])

        all_words = []
        for i in range(row, len(board)):
            for j in range(column, len(board[0])):
                if board[i][j] is not None:
                    # met a top tile of a col.
                    if board[i + 1][j] is not None and board[i - 1][j] is None:
                        word = check_col(i, j, board)
                        if not self.validate_words([word]):
                            return False
                        all_words.append(word)

                    # met a leftest most tile on the board.
                    if board[i][j + 1] is not None and board[i][j - 1] is None:
                        word = check_row(i, j, board)
                        if not self.validate_words([word]):
                            return False
                        all_words.append(word)

        print(all_words)
        return True

    # FIXME/TODO
    """
    We don't really need a model tile's position (center)
    We only really need to know where it's indexed on the game board. 
    
    We can make a hashmap in main.py, where each center of a tile corresponds to 
    and index in the tile. making it so that we don't have to loop through each cell of the board to find collision points.
    """

    def place_tile_on_board(self, tile, center):
        tile.model_tile.set_position(center)
        if tile in self.tiles_on_board:
            self.tiles_on_board.remove(tile)
        self.tiles_on_board.append(tile)

        # remove the tile from the bench. if we take the tile from the bench and
        # place it on the board, we want to remove it from the bench.
        if tile.model_tile in self.tiles_on_bench:
            self.tiles_on_bench.remove(tile.model_tile)

        self.board_valid = self.validate()
        # dumps the coordinates etc into a json file for review.
        self.dump_board()

        if len(self.tiles_on_bench) == 0 and self.board_valid:
            self.peel()

    def dump_board(self):
        with open("board.json", encoding="utf-8", mode="w") as f:
            board = self.board
            for i in range(0, len(board)):
                for j in range(0, len(board[0])):
                    if board[i][j] is not None:
                        tile = board[i][j]
                        f.write(tile.get_value())
                        f.write(" has position: ")
                        f.write(f"{i}, {j}")
                        f.write("\n")

    def init_bench(self, count):
        for i in range(0, count):
            self.peel()

    def remaining_tiles(self):
        return self.tile_bank.get_all_remaining_tiles()

    def peel(self):
        token = self.tile_bank.peel()
        self.tiles_on_bench.append(token)

    def dump(self, token):
        if token in self.tiles_on_bench:
            self.tiles_on_bench.remove(token)
        elif token in self.tiles_on_board:
            self.tiles_on_board.remove(token)
        self.tile_bank.dump(token)
        if self.tile_bank.can_dump():
            for i in range(0, 3):
                peeled_tile = self.tile_bank.peel()
                self.tiles_on_bench.append(peeled_tile)

    def get_game_state(self):
        return {"board_valid": self.board_valid}

    def print_layout(self):
        for row in self.coordinates:
            for col in row:
                col.print_position()

    def build_coordinates(self, coordinates: (int, int, int)):
        """
        builds a coordinate system
        """
        width = coordinates[0]
        height = coordinates[1]
        divider = coordinates[2]
        # divider is something like height / divider to get # of
        # cells in a column or something.
        COLS = math.floor(width // divider)
        ROWS = math.floor(height // divider)
        coordinates = []
        for i in range(ROWS):
            coordinates.append([])
            for j in range(COLS):
                cell_position = (i, j)
                # build a coordinate object
                coordinate = Coordinate(
                    j * divider, i * divider, divider, cell_position
                )
                coordinates[i].append(coordinate)
        self.board = [[None for i in x] for x in coordinates]
        return coordinates

    def build_coordinate_ref(self):
        """
        builds a hashmap of cell centers to their respective index in the self.board
        we need a reference point so that when we pass in a tile in the place_tile_on_board method,
        we can take the tile's location on the board and find the index that we want to place it on
        the self.board.
        """
        hash = {}
        for row in range(0, len(self.coordinates)):
            for col in range(0, len(self.coordinates[0])):
                coordinate = self.coordinates[row][col]
                if coordinate:
                    hash[coordinate.get_center()] = (row, col)
        return hash

    def clean_board(self):
        self.board = [[None for i in x] for x in self.coordinates]


class Coordinate:
    def __init__(self, x, y, divider, position):
        self.x = x
        self.y = y
        self.position = position  # where this is in the matrix. ie (1, 3)
        self.divider = divider

    def get_position_in_grid(self):
        return self.position

    def get_center(self):
        return (self.x + self.divider // 2, self.y + self.divider // 2)

    def print_coordinate(self):
        coord = "X: " + self.x.__str__() + " Y: " + self.y.__str__()
        print(coord)


class ModelTile:
    def __init__(self, value: str, position):
        self.value = value
        self.position = position
        self.id = uuid.uuid4().__str__()

    def get_value(self) -> str:
        return self.value

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    # NOTE, this needs to be in place. Otherwise when doing equality checks
    # within a dictionary (tile in dictionary), we will be performing an equality check
    # for the tiles default .value field, rather than a specific field that we define
    # uniquely per class instance.
    def __eq__(self, other):
        if not isinstance(other, ModelTile):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class TileBank:
    def __init__(self):
        self.bank = init_game_tiles()

    def get_bank_size(self):
        return self.bank.__len__()

    def get_all_remaining_tiles(self):
        return self.bank

    def get_current_size(self):
        return len(self.bank)

    def can_dump(self):
        if len(self.bank) >= 3:
            return True
        return False

    def peel(self) -> ModelTile:
        """
        removes a value from the tile bank and returns it
        """
        index = random.randint(0, len(self.bank) - 1)
        token = self.bank[index]
        self.bank.remove(token)
        return token

    def dump(self, token):
        """
        if there are 3 tiles left in the bank, 3 tiles are selected at random.
        those 3 tiles are returned to the user in exchange for a token provided
        by the user.
        """
        self.bank.append(token)
        random.shuffle(self.bank)


def init_game_tiles():
    bananagrams_tiles = []

    # Create tiles based on letter frequency
    letter_counts = {
        "A": 13,
        "B": 3,
        "C": 3,
        "D": 6,
        "E": 18,
        "F": 3,
        "G": 4,
        "H": 3,
        "I": 12,
        "J": 2,
        "K": 2,
        "L": 5,
        "M": 3,
        "N": 8,
        "O": 11,
        "P": 3,
        "Q": 2,
        "R": 9,
        "S": 6,
        "T": 9,
        "U": 6,
        "V": 3,
        "W": 3,
        "X": 2,
        "Y": 3,
        "Z": 2,
    }

    for letter, count in letter_counts.items():
        for _ in range(count):
            bananagrams_tiles.append(ModelTile(letter, position=(0, 0)))

    random.shuffle(bananagrams_tiles)
    return bananagrams_tiles
