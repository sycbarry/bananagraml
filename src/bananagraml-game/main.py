# Example file showing a circle moving on screen
import pygame
from src.game.model import BananaGramlModel
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional


# Constants
@dataclass
class GameConfig:
    # Screen Dimensions
    SCREEN_WIDTH: int = 1260
    SCREEN_HEIGHT: int = 720

    # Bench Dimensions
    BENCH_HEIGHT: int = 70
    BENCH_WIDTH: int = SCREEN_WIDTH

    # Board Dimensions
    BOARD_HEIGHT: int = 650
    BOARD_WIDTH: int = SCREEN_WIDTH
    DIVIDER: int = 30  # larger # means less tiles in a row or column

    # Colors
    SELECT_COLOR: Tuple[int, int, int, int] = (0, 255, 0, 100)  # Semi-transparent green
    BOX_COLOR: Tuple[int, int, int] = (255, 0, 0)
    FONT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    TILE_COLOR: Tuple[int, int, int] = (255, 239, 184)
    TILE_HOVER_COLOR: Tuple[int, int, int] = (240, 220, 170)
    TILE_SELECTED_COLOR: Tuple[int, int, int] = (200, 255, 200)
    CELL_COLOR: Tuple[int, int, int] = (30, 30, 30)
    CELL_HOVER_COLOR: Tuple[int, int, int] = (20, 20, 20)
    BOARD_COLOR: str = "blue"
    BENCH_COLOR: str = "green"
    BACKGROUND_COLOR: Tuple[int, int, int] = (40, 44, 52)
    DUMP_AREA_COLOR: Tuple[int, int, int] = (255, 0, 0)  # Red color for dump area

    # Dump Area Dimensions
    DUMP_AREA_SIZE: int = 50
    DUMP_AREA_MARGIN: int = 20


class Cell(pygame.sprite.Sprite):
    def __init__(
        self,
        pos: Tuple[int, int],
        size: Tuple[int, int],
        coordinate_object: Tuple[int, int],
    ):
        super().__init__()
        self.image = pygame.Surface(size)
        self.coordinate_object = coordinate_object
        self.original_color = GameConfig.CELL_COLOR
        self.image.fill(self.original_color)
        self.rect = self.image.get_rect(center=pos)
        self._render_text()
        self._render_center()

    def update(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEMOTION:
            self.image.fill(
                GameConfig.CELL_HOVER_COLOR
                if self.rect.collidepoint(event.pos)
                else self.original_color
            )
            # self._render_text()  # Re-render text after filling
            self._render_center()

    def _render_text(self) -> None:
        font = pygame.font.Font(None, 13)
        text = f"{self.rect.centerx}, {self.rect.centery}"
        text_surface = font.render(text, True, "white")
        text_rect = text_surface.get_rect(
            center=(self.image.get_width() // 2, self.image.get_height() // 2)
        )
        self.image.blit(text_surface, text_rect)

    def _render_center(self) -> None:
        pygame.draw.circle(
            self.image,
            "black",
            center=(self.image.get_width() // 2, self.image.get_height() // 2),
            radius=4,
        )


class Tile(pygame.sprite.Sprite):
    def __init__(self, pos: Tuple[int, int], size: Tuple[int, int], model_tile=None):
        super().__init__()
        self.image = pygame.Surface(size)
        self.original_color = GameConfig.TILE_COLOR
        self.image.fill(self.original_color)
        self.rect = self.image.get_rect(center=pos)
        self.dragging = False
        self.model_tile = model_tile
        self.offset = pygame.math.Vector2(0, 0)
        self.original_position = pos
        self.is_selected = False
        self._render_text()

    def _render_text(self) -> None:
        if self.model_tile and self.model_tile.value:
            font = pygame.font.Font(None, 24)
            text_surface = font.render(
                self.model_tile.value, True, GameConfig.FONT_COLOR
            )
            text_rect = text_surface.get_rect(
                center=(self.image.get_width() // 2, self.image.get_height() // 2)
            )
            self.image.blit(text_surface, text_rect)

    def update(
        self,
        event: pygame.event.Event,
        cells: pygame.sprite.Group,
        model: BananaGramlModel,
    ) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and not self.is_selected:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.offset = pygame.math.Vector2(self.rect.center) - event.pos

        elif event.type == pygame.MOUSEBUTTONUP and self.dragging:
            self.handle_drop(model, cells)

        elif event.type == pygame.MOUSEMOTION:
            self.handle_motion(event, cells)

    def handle_drop(
        self,
        model: BananaGramlModel,
        cells: pygame.sprite.Group,
    ) -> None:
        dump_area = GameRenderer.draw_dump_area()

        # Check if tile was dropped in dump area
        if dump_area.collidepoint(self.rect.center):
            if self.model_tile:
                model.dump(self.model_tile)
                self.dragging = False  # Ensure dragging is set to False
                self.kill()  # Remove the tile sprite from the UI.
                return

        # Check if tile was dropped on another board tile
        for board_tile in model.board_tiles():
            if self == board_tile:
                continue
            if self.rect.collidepoint(board_tile.rect.center):
                self.rect.center = self.original_position
                model.tiles_on_bench.append(self.model_tile)
                self.dragging = False
                return

        # TODO
        # when dropping tile on cell, we need to ensure that the tile is being placed directly
        # on a coordainte that is identical to what we have in our model.coordinate_ref hash
        for cell in cells:
            if self.rect.collidepoint(cell.rect.center):
                model.place_tile_on_board(self, cell.rect.center)
                self.rect.center = cell.rect.center
                self.original_position = cell.rect.center
                self.dragging = False

    def handle_motion(
        self, event: pygame.event.Event, cells: pygame.sprite.Group
    ) -> None:
        if self.dragging and not self.is_selected:
            self.rect.center = event.pos + self.offset
            # stick to the center of the cell that the tile is being hovered over.
            for cell in cells:
                if self.rect.collidepoint(cell.rect.center):
                    self.rect.center = cell.rect.center
        self._update_appearance(event.pos)

    def change_background_color(self, isvalid: bool):
        if not isvalid:
            color = "red"
        else:
            # color = self.original_color
            color = "green"
        self.image.fill(color)
        self._render_text()

    def _update_appearance(self, mouse_pos: Tuple[int, int]) -> None:
        if self.is_selected:
            color = GameConfig.TILE_SELECTED_COLOR
        elif self.rect.collidepoint(mouse_pos):
            color = GameConfig.TILE_HOVER_COLOR
        else:
            color = self.original_color

        self.image.fill(color)
        self._render_text()


class DragSelect:
    def __init__(self):
        self.selected_tiles: List[Tile] = []
        self.selecting: bool = False
        self.start_pos: Tuple[int, int] = (0, 0)
        self.selection_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self.dragging_group: bool = False
        self.group_offset: Optional[List[pygame.math.Vector2]] = None

    def clear_selection(self) -> None:
        for tile in self.selected_tiles:
            tile.is_selected = False
        self.selected_tiles = []
        self.selecting = False
        self.dragging_group = False
        self.group_offset = None

    def start_selection(self, pos: Tuple[int, int]) -> None:
        self.clear_selection()
        self.selecting = True
        self.start_pos = pos
        self.selection_rect = pygame.Rect(*pos, 0, 0)

    def update_selection(self, pos: Tuple[int, int], tiles: List[Tile]) -> None:
        if not self.selecting:
            return

        self.selected_tiles = []
        self._update_selection_rect(pos)

        for tile in tiles:
            tile.is_selected = self.selection_rect.collidepoint(tile.rect.center)
            if tile.is_selected:
                self.selected_tiles.append(tile)

    def _update_selection_rect(self, pos: Tuple[int, int]) -> None:
        width = pos[0] - self.start_pos[0]
        height = pos[1] - self.start_pos[1]

        self.selection_rect.x = pos[0] if width < 0 else self.start_pos[0]
        self.selection_rect.width = abs(width)
        self.selection_rect.y = pos[1] if height < 0 else self.start_pos[1]
        self.selection_rect.height = abs(height)

    def start_group_drag(self, pos: Tuple[int, int], clicked_tile: Tile) -> None:
        if clicked_tile in self.selected_tiles:
            self.dragging_group = True
            clicked_center = pygame.math.Vector2(clicked_tile.rect.center)
            self.group_offset = [
                pygame.math.Vector2(tile.rect.center) - clicked_center
                for tile in self.selected_tiles
            ]

    def update_group_drag(
        self, pos: Tuple[int, int], cells: pygame.sprite.Group
    ) -> None:
        if not (self.dragging_group and self.group_offset):
            return

        base_pos = pygame.math.Vector2(pos)
        self._update_tile_positions(base_pos, cells)

    def _update_tile_positions(
        self, base_pos: pygame.math.Vector2, cells: pygame.sprite.Group
    ) -> None:
        # First update all positions
        for tile, offset in zip(self.selected_tiles, self.group_offset):
            tile.rect.center = base_pos + offset

            # Check for cell snapping
            for cell in cells:
                if tile.rect.collidepoint(cell.rect.center):
                    snap_adjustment = pygame.math.Vector2(cell.rect.center) - (
                        base_pos + offset
                    )
                    base_pos += snap_adjustment
                    break

        # Update all positions again with snapped base_pos
        for tile, offset in zip(self.selected_tiles, self.group_offset):
            tile.rect.center = base_pos + offset

    def end_group_drag(
        self,
        model: BananaGramlModel,
        cells: pygame.sprite.Group,
    ) -> None:
        if self.dragging_group:
            self.dragging_group = False
            self.group_offset = None
            for tile in self.selected_tiles:
                model.place_tile_on_board(tile, tile.rect.center)

    def end_selection(self) -> None:
        self.selecting = False

    def draw(self, surface: pygame.Surface, tiles: List[Tile]) -> None:
        if self.selecting:
            pygame.draw.rect(
                surface, GameConfig.SELECT_COLOR[:3], self.selection_rect, 2
            )


class GameRenderer:
    @staticmethod
    def create_cells(model: BananaGramlModel) -> pygame.sprite.Group:
        """
        this creates the visual sprites on the board with their
        respective centers
        """
        cells = pygame.sprite.Group()
        for row in model.coordinates:
            size = GameConfig.DIVIDER
            for col in row:
                cell = Cell(
                    pos=col.get_center(),
                    size=(size, size),
                    coordinate_object=col,  # this is the coordinate point
                )
                cells.add(cell)
        return cells

    @staticmethod
    def draw_stats_area(screen, model) -> pygame.Rect:
        x = GameConfig.SCREEN_WIDTH - 100 - 150
        y = (
            GameConfig.SCREEN_HEIGHT
            - GameConfig.DUMP_AREA_SIZE
            - GameConfig.DUMP_AREA_MARGIN
        )
        rect = pygame.Rect(x, y, 180, 50)
        font = pygame.font.Font(None, 20)
        final_text = "".join(
            [
                f"tiles on board: {len(model.tiles_on_board)}\n"
                f"tiles on bench: {len(model.tiles_on_bench)}\n"
                f"tiles in bank: {model.tile_bank.get_bank_size()}\n"
            ]
        )
        text_surface = font.render(final_text, True, "white")
        text_rect = text_surface.get_rect(center=(rect.center))
        screen.blit(text_surface, text_rect)
        return rect

    @staticmethod
    def draw_dump_area() -> pygame.Rect:
        x = (
            GameConfig.SCREEN_WIDTH
            - GameConfig.DUMP_AREA_SIZE
            - GameConfig.DUMP_AREA_MARGIN
        )
        y = (
            GameConfig.SCREEN_HEIGHT
            - GameConfig.DUMP_AREA_SIZE
            - GameConfig.DUMP_AREA_MARGIN
        )
        return pygame.Rect(x, y, GameConfig.DUMP_AREA_SIZE, GameConfig.DUMP_AREA_SIZE)

    @staticmethod
    def render_bench_tiles(
        model: BananaGramlModel, bench: pygame.Surface, existing_tiles=None
    ) -> pygame.sprite.Group:
        tiles = pygame.sprite.Group()
        if not model.tiles_on_bench:
            return tiles
        X_orig = bench.get_rect().x + 20
        Y_orig = GameConfig.SCREEN_HEIGHT - GameConfig.BENCH_HEIGHT + 50

        # Track which model tiles have been handled
        handled_model_tiles = []
        
        # First, preserve existing tiles that are in the bench
        if existing_tiles:
            for tile in existing_tiles:
                if tile.model_tile in model.tiles_on_bench:
                    # Keep the existing tile in the group
                    tiles.add(tile)
                    handled_model_tiles.append(tile.model_tile)
                    
                    # Only reset position if it's not being dragged
                    if not tile.dragging:
                        if tile.rect.center != tile.original_position:
                            tile.rect.center = (X_orig, Y_orig)
                    X_orig += 25
        
        # Create new tiles for any model tiles that weren't in existing_tiles
        for model_tile in model.tiles_on_bench:
            if model_tile not in handled_model_tiles:
                game_tile = Tile(
                    pos=(X_orig, Y_orig), size=(20, 20), model_tile=model_tile
                )
                tiles.add(game_tile)
                X_orig += 25
                
        return tiles

    @staticmethod
    def draw_board() -> pygame.Surface:
        board = pygame.Surface((GameConfig.BOARD_WIDTH, GameConfig.BOARD_HEIGHT))
        board.fill(GameConfig.BOARD_COLOR)
        return board

    @staticmethod
    def draw_bench() -> pygame.Surface:
        bench = pygame.Surface((GameConfig.BENCH_WIDTH, GameConfig.BENCH_HEIGHT))
        bench.fill(GameConfig.BENCH_COLOR)
        return bench


class Game:
    def __init__(self, model: BananaGramlModel):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT)
        )
        self.clock = pygame.time.Clock()
        self.model = model
        self.drag_select = DragSelect()
        self.is_dragging = False
        self.board_cells = GameRenderer.create_cells(model)
        self.bench = GameRenderer.draw_bench()
        self.board = GameRenderer.draw_board()
        self.bench_tiles = GameRenderer.render_bench_tiles(model, self.bench)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            self.board_cells.update(event)
            self._update_tiles(event)
            self._handle_mouse_event(event)
        return True

    def _update_tiles(self, event: pygame.event.Event) -> None:
        for tile in self.model.tiles_on_board:
            if isinstance(tile, Tile):
                tile.update(event, self.board_cells, self.model)
        if self.bench_tiles:
            self.bench_tiles.update(event, self.board_cells, self.model)

    def _handle_mouse_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(event)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._handle_mouse_up()

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        clicked_tile = next(
            (
                tile
                for tile in self.drag_select.selected_tiles
                if tile.rect.collidepoint(event.pos)
            ),
            None,
        )

        if clicked_tile:
            self.drag_select.start_group_drag(event.pos, clicked_tile)
            return

        clicked_any_tile = any(
            tile.rect.collidepoint(event.pos) for tile in self._get_all_tiles()
        )

        if (
            event.pos[1] < GameConfig.SCREEN_HEIGHT - GameConfig.BENCH_HEIGHT
            and not clicked_any_tile
        ):
            self.drag_select.start_selection(event.pos)
        else:
            self.drag_select.clear_selection()

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        if self.drag_select.dragging_group:
            self.drag_select.update_group_drag(event.pos, self.board_cells)
        elif self.drag_select.selecting:
            self.drag_select.update_selection(event.pos, self._get_all_tiles())

    def _handle_mouse_up(self) -> None:
        if self.drag_select.dragging_group:
            self.drag_select.end_group_drag(self.model, self.board_cells)
        elif self.drag_select.selecting:
            self.drag_select.end_selection()

    def _get_all_tiles(self) -> List[Tile]:
        board_tiles = [t for t in self.model.tiles_on_board if isinstance(t, Tile)]
        bench_tiles_list = list(self.bench_tiles) if self.bench_tiles else []
        return board_tiles + bench_tiles_list

    def render(self) -> None:
        self.screen.fill(GameConfig.BACKGROUND_COLOR)
        self.board_cells.draw(self.screen)

        # Draw board tiles
        for tile in self.model.tiles_on_board:
            if isinstance(tile, Tile):
                tile.change_background_color(self.model.board_valid)
                self.screen.blit(tile.image, tile.rect)

        # draw the bench tiles on the bench
        self.bench_tiles = GameRenderer.render_bench_tiles(
            self.model, self.bench, self.bench_tiles
        )
        self.bench_tiles.draw(self.screen)

        # render dump area
        dump_area = GameRenderer.draw_dump_area()
        pygame.draw.rect(self.screen, GameConfig.DUMP_AREA_COLOR, dump_area, 2)
        stats = GameRenderer.draw_stats_area(self.screen, self.model)
        pygame.draw.rect(self.screen, "blue", stats, 2)
        self.drag_select.draw(self.screen, self.model.tiles_on_board)
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.render()
            self.clock.tick(60)
        pygame.quit()


def main():
    board_dimensions = (
        # something like 1080*720
        # where 1080 is width and 720 is height
        GameConfig.BOARD_WIDTH,  # is longer than height
        GameConfig.BOARD_HEIGHT,  # is shorter than width.
        GameConfig.DIVIDER,
    )
    model = BananaGramlModel(board_dimensions)
    model.init_bench(10)
    game = Game(model)
    game.run()


if __name__ == "__main__":
    main()
