"""
- this is the data model that we adjust and use to 
  verify the game's current state.
"""
import math
import random


class BananaGramlModel:
    def __init__(self, BOARD_DIMENSIONS: (int, int, int)):
        self.board_valid = True
        self.coordinates = build_coordinates(BOARD_DIMENSIONS)
        self.tile_bank = TileBank()
        self.tiles_on_board = []
        self.tiles_on_bench = []

    def board_tiles(self):
        return self.tiles_on_board

    """
    def place_tile_on_board(self, tile, position):
        if tile in self.tiles_on_board:
            self.tiles_on_board.remove(tile)
        tile.set_position(position)
        self.tiles_on_board.append(tile)
    """

    def place_tile_on_board(self, tile):
        if tile in self.tiles_on_board:
            self.tiles_on_board.remove(tile)
        self.tiles_on_board.append(tile)
        if tile.model_tile in self.tiles_on_bench:
            self.tiles_on_bench.remove(tile.model_tile)
        if len(self.tiles_on_bench) == 0:
            self.peel()

    def init_bench(self, count):
        # for i in range(0, 20):
        for i in range(0, count):
            self.peel()

    def remaining_tiles(self):
        return self.tile_bank.get_all_remaining_tiles()

    def peel(self):
        token = self.tile_bank.peel()
        self.tiles_on_bench.append(token)

    def dump(self, token, location="bench"):
        if location == "bench":
            self.tiles_on_bench.remove(token)
        else:
            self.tiles_on_board.remove(token)
        new_tokens = self.tile_bank.dump(token)
        for token in new_tokens:
            self.tiles_on_bench.append(token)

    def get_game_state(self):
        return {"board_valid": self.board_valid}

    def print_layout(self):
        for row in self.coordinates:
            for col in row:
                col.print_position()


class Coordinate:
    def __init__(self, x, y, divider, position):
        self.x = x
        self.y = y
        self.position = position
        self.divider = divider

    def get_center(self):
        return (self.x + self.divider / 2, self.y + self.divider / 2)

    def print_coordinate(self):
        coord = "X: " + self.x.__str__() + " Y: " + self.y.__str__()
        print(coord)


class ModelTile:
    def __init__(self, value: str, position):
        self.value = value
        self.position = position

    def __str__(self):
        return self.value

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position


class TileBank:
    def __init__(self):
        self.bank = init_game_tiles()

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
        index = random.randint(0, len(self.bank))
        token = self.bank[index]
        self.bank.pop(index)
        return token

    def dump(self, token):
        """
        if there are 3 tiles left in the bank, 3 tiles are selected at random.
        those 3 tiles are returned to the user in exchange for a token provided
        by the user.
        """
        if len(self.bank) >= 3:
            self.bank.append(token)
            returned_tokens = []
            for i in range(0, 2):
                returned_tokens.append(self.peel())
            return tuple(returned_tokens)
        else:
            # TODO provide some better ruling logic on how to handle this case.
            return token


def init_game_tiles():
    bananagrams_tiles = (
        [ModelTile("A", position=(0, 0))] * 13
        + [ModelTile("B", position=(0, 0))] * 3
        + [ModelTile("C", position=(0, 0))] * 3
        + [ModelTile("D", position=(0, 0))] * 6
        + [ModelTile("E", position=(0, 0))] * 18
        + [ModelTile("F", position=(0, 0))] * 3
        + [ModelTile("G", position=(0, 0))] * 4
        + [ModelTile("H", position=(0, 0))] * 3
        + [ModelTile("I", position=(0, 0))] * 12
        + [ModelTile("J", position=(0, 0))] * 2
        + [ModelTile("K", position=(0, 0))] * 2
        + [ModelTile("L", position=(0, 0))] * 5
        + [ModelTile("M", position=(0, 0))] * 3
        + [ModelTile("N", position=(0, 0))] * 8
        + [ModelTile("O", position=(0, 0))] * 11
        + [ModelTile("P", position=(0, 0))] * 3
        + [ModelTile("Q", position=(0, 0))] * 2
        + [ModelTile("R", position=(0, 0))] * 9
        + [ModelTile("S", position=(0, 0))] * 6
        + [ModelTile("T", position=(0, 0))] * 9
        + [ModelTile("U", position=(0, 0))] * 6
        + [ModelTile("V", position=(0, 0))] * 3
        + [ModelTile("W", position=(0, 0))] * 3
        + [ModelTile("X", position=(0, 0))] * 2
        + [ModelTile("Y", position=(0, 0))] * 3
        + [ModelTile("Z", position=(0, 0))] * 2
    )
    random.shuffle(bananagrams_tiles)
    return bananagrams_tiles


def build_coordinates(coordinates: (int, int, int)):
    """
    builds a coordinate system
    """
    X = coordinates[0]
    Y = coordinates[1]
    divider = coordinates[2]
    COLS = math.floor(X / divider)
    ROWS = math.floor(Y / divider)
    coordinates = []
    for i in range(ROWS):
        coordinates.append([])
        for j in range(COLS):
            coordinates[i].append(Coordinate(i * divider, j * divider, divider, (i, j)))

    return coordinates
