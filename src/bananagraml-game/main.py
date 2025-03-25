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
        self.dragging_group = False
        self.group_offset = None

    def clear_selection(self):
        """Clear all selected tiles"""
        for tile in self.selected_tiles:
            tile.is_selected = False
        self.selected_tiles = []
        self.selecting = False
        self.dragging_group = False
        self.group_offset = None

    def start_selection(self, pos):
        # Clear previous selection when starting a new one
        self.clear_selection()
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
                    tile.is_selected = True
                else:
                    tile.is_selected = False

    def start_group_drag(self, pos, clicked_tile):
        """Start dragging the group from a clicked tile"""
        if clicked_tile in self.selected_tiles:
            self.dragging_group = True
            # Calculate offsets for all tiles relative to clicked tile
            clicked_center = pygame.math.Vector2(clicked_tile.rect.center)
            self.group_offset = []
            for tile in self.selected_tiles:
                offset = pygame.math.Vector2(tile.rect.center) - clicked_center
                self.group_offset.append(offset)

    def update_group_drag(self, pos, cells):
        """Update positions of all selected tiles during group drag"""
        if self.dragging_group and self.group_offset:
            base_pos = pygame.math.Vector2(pos)

            # First update all positions
            for tile, offset in zip(self.selected_tiles, self.group_offset):
                tile.rect.center = base_pos + offset

                # Check for cell snapping
                for cell in cells:
                    if tile.rect.collidepoint(cell.rect.center):
                        # Adjust base_pos to account for snapping
                        snap_adjustment = pygame.math.Vector2(cell.rect.center) - (
                            base_pos + offset
                        )
                        base_pos += snap_adjustment
                        break

            # Update all positions again with snapped base_pos
            for tile, offset in zip(self.selected_tiles, self.group_offset):
                tile.rect.center = base_pos + offset

    def end_group_drag(self, model):
        """End group dragging and update model positions"""
        if self.dragging_group:
            self.dragging_group = False
            self.group_offset = None
            # Update model positions for all tiles
            for tile in self.selected_tiles:
                model.place_tile_on_board(tile)

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
        self.is_selected = False

        # Render text
        if model_tile and model_tile.value:
            text_surface = font.render(model_tile.value, True, FONT_COLOR)
            text_rect = text_surface.get_rect(center=(size[0] // 2, size[1] // 2))
            self.image.blit(text_surface, text_rect)

    def update(self, event, cells, model):
        """Handle both individual tile dragging and selection states"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos) and not self.is_selected:
                # Only start individual dragging if tile is not part of a selection
                self.dragging = True
                self.offset = pygame.math.Vector2(self.rect.center) - event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if self.dragging:
                # check if we have a tile position conflict
                for board_tile in model.board_tiles():
                    if self == board_tile:
                        continue
                    if self.rect.collidepoint(board_tile.rect.center):
                        self.rect.center = self.original_position
                        self.dragging = False
                        return
                self.dragging = False
                model.place_tile_on_board(self)
                self.original_position = self.rect.center

        elif event.type == pygame.MOUSEMOTION:
            # Handle dragging
            if self.dragging and not self.is_selected:
                self.rect.center = event.pos + self.offset
                for cell in cells:
                    if self.rect.collidepoint(cell.rect.center):
                        self.rect.center = cell.rect.center

            # Update appearance based on state
            if self.is_selected:
                self.image.fill((200, 255, 200))  # Light green for selected
            elif self.rect.collidepoint(event.pos):
                self.image.fill((240, 220, 170))  # Hover color
            else:
                self.image.fill(self.original_color)

            # Re-render text
            if self.model_tile and self.model_tile.value:
                text_surface = font.render(self.model_tile.value, True, FONT_COLOR)
                text_rect = text_surface.get_rect(
                    center=(self.image.get_width() // 2, self.image.get_height() // 2)
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


def render_bench_tiles(model, bench, existing_tiles=None):
    """Create sprite group for bench tiles with current state"""
    tiles = pygame.sprite.Group()
    if model.tiles_on_bench:
        X_orig = bench.get_rect().x + 20
        Y_orig = SCREEN_HEIGHT - BENCH_HEIGHT + 50
        
        # Create a map of existing tiles by their model_tile
        existing_tile_map = {}
        if existing_tiles:
            for tile in existing_tiles:
                if tile.model_tile and tile.model_tile in model.tiles_on_bench:  # Only map tiles that are still on bench
                    existing_tile_map[tile.model_tile] = tile

        for model_tile in model.tiles_on_bench:
            # If we have an existing tile for this model, use it
            if model_tile in existing_tile_map:
                game_tile = existing_tile_map[model_tile]
                # Update position if not being dragged
                if not game_tile.dragging:
                    game_tile.rect.center = (X_orig, Y_orig)
                tiles.add(game_tile)
            else:
                # Create new tile if we don't have one
                game_tile = Tile(
                    size=((20, 20)), color="blue", pos=(X_orig, Y_orig), model_tile=model_tile
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
    bench_tiles = render_bench_tiles(game_model, bench)

    while running:
        screen.fill((40, 44, 52))
        board_cells.draw(screen)
        
        # Draw board tiles first
        for tile in game_model.tiles_on_board:
            if isinstance(tile, Tile):  # If it's already a sprite
                screen.blit(tile.image, tile.rect)
        
        # Re-render bench tiles every frame to catch any updates, but maintain state
        bench_tiles = render_bench_tiles(game_model, bench, bench_tiles)
        if bench_tiles:
            bench_tiles.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            board_cells.update(event)
            
            # Update both board and bench tiles
            for tile in game_model.tiles_on_board:
                if isinstance(tile, Tile):
                    tile.update(event, board_cells, game_model)
            if bench_tiles:
                bench_tiles.update(event, board_cells, game_model)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check if clicking on a selected tile first
                    clicked_tile = None
                    for tile in drag_select.selected_tiles:
                        if tile.rect.collidepoint(event.pos):
                            clicked_tile = tile
                            break

                    if clicked_tile:
                        # Start group dragging
                        drag_select.start_group_drag(event.pos, clicked_tile)
                    else:
                        # Check if clicking on any tile (board or bench)
                        clicked_any_tile = False
                        all_tiles = [t for t in game_model.tiles_on_board if isinstance(t, Tile)] + (list(bench_tiles) if bench_tiles else [])
                        for tile in all_tiles:
                            if tile.rect.collidepoint(event.pos):
                                clicked_any_tile = True
                                break

                        # Only start selection if we're on the board area AND not clicking any tile
                        if event.pos[1] < SCREEN_HEIGHT - BENCH_HEIGHT and not clicked_any_tile:
                            drag_select.start_selection(event.pos)
                        elif clicked_any_tile:
                            # Clear selection if clicking a tile
                            drag_select.clear_selection()
                        else:
                            # Clicked in bench area, clear selection
                            drag_select.clear_selection()

            elif event.type == pygame.MOUSEMOTION:
                if drag_select.dragging_group:
                    drag_select.update_group_drag(event.pos, board_cells)
                elif drag_select.selecting:
                    # Update selection with all tiles
                    all_tiles = [t for t in game_model.tiles_on_board if isinstance(t, Tile)] + (list(bench_tiles) if bench_tiles else [])
                    drag_select.update_selection(event.pos, all_tiles)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if drag_select.dragging_group:
                        drag_select.end_group_drag(game_model)
                    elif drag_select.selecting:
                        drag_select.end_selection()

        drag_select.draw(screen, game_model.tiles_on_board)
        pygame.display.flip()
        dt = clock.tick(60) / 1000

    pygame.quit()


model = BananaGramlModel(BOARD_DIMENSIONS)
model.init_bench(1)

if __name__ == "__main__":
    main(model)
