# Example file showing a circle moving on screen
import pygame
from src.game.model import BananaGramlModel
import sys

# Screen Dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800

# Bench Dimensions
BENCH_HEIGHT = 150
BENCH_WIDTH = SCREEN_WIDTH

# Board Dimensions
BOARD_HEIGHT = 650
BOARD_WIDTH = SCREEN_WIDTH
DIVIDER = 30  # larger # means less tiles in a row or column.
BOARD_DIMENSIONS = (BOARD_HEIGHT, BOARD_WIDTH, DIVIDER)

# Colors
SELECT_COLOR = (0, 255, 0, 100)  # Semi-transparent green
BOX_COLOR = (255, 0, 0)
FONT_COLOR = (0, 0, 0)

# Initialize pygame and font
pygame.init()
font = pygame.font.Font(None, 24)  # None uses default font, 24 is the size


class DragSelect:
    def __init__(self):
        self.selected_tiles = []
        self.selecting = False
        self.start_pos = (0, 0)
        self.selection_rect = pygame.Rect(0, 0, 0, 0)

    def start_selection(self, pos):
        self.selecting = True
        self.start_pos = pos
        self.selection_rect = pygame.Rect(*pos, 0, 0)

    def update_selection(self, pos, tiles):
        if self.selecting:
            self.selected_tiles = []

            # Calculate width and height
            width = pos[0] - self.start_pos[0]
            height = pos[1] - self.start_pos[1]

            # Normalize the rectangle
            if width < 0:
                self.selection_rect.x = pos[0]
                self.selection_rect.width = abs(width)
            else:
                self.selection_rect.x = self.start_pos[0]
                self.selection_rect.width = width

            if height < 0:
                self.selection_rect.y = pos[1]
                self.selection_rect.height = abs(height)
            else:
                self.selection_rect.y = self.start_pos[1]
                self.selection_rect.height = height

            for tile in tiles:
                if self.selection_rect.collidepoint(tile.rect.center):
                    self.selected_tiles.append(tile)

    def end_selection(self):
        if self.selecting:
            self.selecting = False

    def draw(self, surface, tiles):
        if self.selecting:
            pygame.draw.rect(
                surface, SELECT_COLOR[:3], self.selection_rect, 2
            )  # Outline

    def get_selected_objects(self, objects):
        return self.selected_objects


class Cell(pygame.sprite.Sprite):
    def __init__(self, color="red", pos=(0, 0), size=(10, 10)):
        color = (30, 30, 30)
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface(size)
        self.original_color = color
        self.image.fill(color)
        self.rect = self.image.get_rect(center=pos)

    def update(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                self.image.fill((20, 20, 20))
            else:
                self.image.fill(self.original_color)


class Tile(pygame.sprite.Sprite):
    def __init__(
        self, color="red", pos=(0, 0), size=(10, 10), text="", model_tile=None
    ):
        color = (255, 239, 184)
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface(size)
        self.original_color = color
        self.image.fill(color)
        self.rect = self.image.get_rect(center=pos)
        self.dragging = False
        self.model_tile = model_tile
        self.offset_x = 0
        self.offset_y = 0
        self.offset = pygame.math.Vector2(0, 0)
        self.original_position = pos

        # Render text
        if model_tile and model_tile.value:
            text_surface = font.render(model_tile.value, True, FONT_COLOR)
            text_rect = text_surface.get_rect(center=(size[0] // 2, size[1] // 2))
            self.image.blit(text_surface, text_rect)

    def reset_color(self):
        self.image.fill(original_color)

    def update_color(self, color):
        self.image.fill(color)

    def update(self, event, cells, model):
        """
        when the user places a tile on a board-cell,
        we need to ensure that we are updating the game model
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.offset = pygame.math.Vector2(self.rect.center) - event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging == True:
                # check if we have a tile position conflict.
                for board_tile in model.board_tiles():
                    if self == board_tile:
                        continue
                    if self.rect.collidepoint(board_tile.rect.center):
                        self.rect.center = self.original_position
                        self.dragging = False
                        return
                self.dragging = False
                # model.place_tile_on_board(self.model_tile, position=self.rect.center)
                model.place_tile_on_board(self)
                self.original_position = self.rect.center

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.rect.center = event.pos + self.offset
                for cell in cells:
                    if self.rect.collidepoint(cell.rect.center):
                        self.rect.center = cell.rect.center
            if self.rect.collidepoint(event.pos):
                self.image.fill((240, 220, 170))
                # Re-render text when hovering
                if self.model_tile and self.model_tile.value:
                    text_surface = font.render(self.model_tile.value, True, FONT_COLOR)
                    text_rect = text_surface.get_rect(
                        center=(
                            self.image.get_width() // 2,
                            self.image.get_height() // 2,
                        )
                    )
                    self.image.blit(text_surface, text_rect)

            else:
                self.image.fill(self.original_color)
                if self.model_tile and self.model_tile.value:
                    text_surface = font.render(self.model_tile.value, True, FONT_COLOR)
                    text_rect = text_surface.get_rect(
                        center=(
                            self.image.get_width() // 2,
                            self.image.get_height() // 2,
                        )
                    )
                    self.image.blit(text_surface, text_rect)


def draw_board():
    board = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT))
    board.fill("blue")
    return board


def draw_bench():
    bench = pygame.Surface((BENCH_WIDTH, BENCH_HEIGHT))
    bench.fill("green")
    return bench


def create_cells(model):
    """Create all cells once and return them as a sprite group."""
    cells = pygame.sprite.Group()
    for row in model.coordinates:
        for col in row:
            cell = Cell(pos=col.get_center(), size=(len(row), len(row)))
            cells.add(cell)
    return cells


def create_bench_tiles(model, bench):
    tiles = pygame.sprite.Group()
    X_orig = bench.get_rect().x + 20
    Y_orig = SCREEN_HEIGHT - BENCH_HEIGHT + 50
    for tile in model.tiles_on_bench:
        game_tile = Tile(
            size=((20, 20)), color="blue", pos=(X_orig, Y_orig), model_tile=tile
        )
        tiles.add(game_tile)
        X_orig += 25
    return tiles


objects = []
selected_tiles = []
tiles_on_board = []


def main(game_model):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill(color=pygame.color.Color(200, 200, 200))
    clock = pygame.time.Clock()
    running = True
    dt = 0
    board_cells = create_cells(game_model)
    drag_select = DragSelect()
    bench = draw_bench()
    board = draw_board()
    screen.blit(bench, dest=(0, SCREEN_HEIGHT - BENCH_HEIGHT))
    screen.blit(board, dest=(0, 0))
    bench_tiles = create_bench_tiles(game_model, bench)

    while running:
        screen.fill((40, 44, 52))
        board_cells.draw(screen)
        bench_tiles.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            board_cells.update(event)
            bench_tiles.update(event, board_cells, model)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drag_select.start_selection(event.pos)
            elif event.type == pygame.MOUSEMOTION:
                drag_select.update_selection(event.pos, model.tiles_on_board)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drag_select.end_selection()

            # drag_select.update(event)

        for obj in objects:
            pygame.draw.rect(screen, BOX_COLOR, obj)
        drag_select.draw(screen, model.tiles_on_board)

        for obj in drag_select.selected_tiles:
            obj.update_color("blue")

        pygame.display.flip()
        pygame.display.update()
        dt = clock.tick(60) / 1000

    pygame.quit()


model = BananaGramlModel(BOARD_DIMENSIONS)
model.init_bench()

if __name__ == "__main__":
    main(model)
