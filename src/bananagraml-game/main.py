# Example file showing a circle moving on screen
import pygame
from src.game.model import BananaGramlModel
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional
import uuid


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
        self.is_focused = False
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

        elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
            self.handle_drop(model, cells)

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

    def change_background_color(self, isvalid: bool, victory: bool):
        if not isvalid:
            color = "red"
        else:
            if victory == True:
                color = "yellow"
            else:
                color = "green"
        self.image.fill(color)
        self._render_text()

    def _update_appearance(self, mouse_pos: Tuple[int, int]) -> None:
        if self.is_selected:
            color = GameConfig.TILE_SELECTED_COLOR
        elif self.rect.collidepoint(mouse_pos):
            print(self.rect.center)
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

        # check for tiles that have already been rendered on the bench
        # so that we are not re-rendering tiles with a static fixed x,y position.
        # so here we create a map of tiles that already exist on the board.
        existing_tile_map = {}
        if existing_tiles:
            for tile in existing_tiles:
                if tile.model_tile in model.tiles_on_bench:
                    existing_tile_map[tile.model_tile] = tile

        # then here we loop through each model tile on the model.tiles_on_bench, and check
        # 1. is the tile already rendered as a sprite? if so, only set it's center to the right of previous tiles on the bench if we are not dragging it
        # 2. if it's not a tile that has been rendered as a sprite, then make a new Tile() object and set it's coordinate center to the right of the previously
        # rendered tiles on the bench.
        for model_tile in model.tiles_on_bench:
            if model_tile in existing_tile_map:
                tile = existing_tile_map[model_tile]
                if not tile.dragging:
                    tile.rect.center = (X_orig, Y_orig)
                tiles.add(tile)
            else:
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

    @staticmethod
    def draw_cross_hair(position, height, width) -> pygame.Rect:
        crosshair = pygame.Rect(position[0], position[1], height, width)
        return crosshair


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
        self.focus_area = "BOARD"
        self.cross_hair_position = (0, 0)  # the default position for the crosshair
        self.bench_cross_hair_position_index = 0
        self.bench_cross_hair_position = (
            100,
            100,
        )  # the default position for the crosshair
        self.cross_hair = GameRenderer.draw_cross_hair(self.cross_hair_position, 30, 30)
        self.board_cells = GameRenderer.create_cells(model)
        self.bench = GameRenderer.draw_bench()
        self.board = GameRenderer.draw_board()
        self.bench_tiles = GameRenderer.render_bench_tiles(model, self.bench)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            self._handle_keyboard_actions(event)
            self.board_cells.update(event)
            self._update_tiles(event)
            self._handle_mouse_event(event)
        return True

    def _handle_board_keyboard_actions(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                if self.selected_tile is not None:
                    for cell in self.board_cells:
                        # TODO
                        """
                        - ensure we are not handle_drop in the tile everytime we hit enter.
                        - we need to make sure that we can hit "enter" on a tile on the board and 
                            move it around as well 
                            (
                                - ie same functionality as the the bench to tile
                                - needs to be in place for board tile to board cell
                            )
                        """
                        if self.cross_hair.collidepoint(cell.rect.center):
                            self.selected_tile.rect.center = cell.rect.center
                            self.selected_tile = None
                else:
                    if self.focus_area == "BOARD":
                        for tile in self.model.board_tiles():
                            if self.cross_hair.collidepoint(tile.rect.center):
                                self.selected_tile = tile
            if event.key == pygame.K_z:
                if self.focus_area == "BENCH":
                    self.focus_area = "BOARD"
                else:
                    self.focus_area = "BENCH"
            # left arrow
            if event.key == pygame.K_LEFT:
                if self.cross_hair_position[0] < 30:
                    return self.cross_hair_position
                c = list(self.cross_hair_position)
                c[0] = c[0] - 30
                self.cross_hair_position = tuple(c)
            # right arrow
            if event.key == pygame.K_RIGHT:
                if (
                    self.cross_hair_position[0]
                    >= pygame.sprite.Group.sprites(self.board_cells)[
                        len(self.board_cells) - 1
                    ].rect.center[0]
                    - 30
                ):
                    return self.cross_hair_position
                c = list(self.cross_hair_position)
                c[0] = c[0] + 30
                self.cross_hair_position = tuple(c)
            # up arrow
            if event.key == pygame.K_UP:
                if self.cross_hair_position[1] < 30:
                    return self.cross_hair_position
                c = list(self.cross_hair_position)
                c[1] = c[1] - 30
                self.cross_hair_position = tuple(c)
            # down arrow
            if event.key == pygame.K_DOWN:
                if (
                    self.cross_hair_position[1]
                    >= pygame.sprite.Group.sprites(self.board_cells)[
                        len(self.board_cells) - 1
                    ].rect.center[1]
                    + 30
                ):
                    return self.cross_hair_position
                c = list(self.cross_hair_position)
                c[1] = c[1] + 30
                self.cross_hair_position = tuple(c)

    def init_bench_cross_hair(self):
        bench_tiles = self.bench_tiles.sprites()
        if bench_tiles is not None:
            if len(bench_tiles) > 0:
                print(len(bench_tiles), self.bench_cross_hair_position_index)
                x = bench_tiles[self.bench_cross_hair_position_index].rect.x
                y = bench_tiles[self.bench_cross_hair_position_index].rect.y
                first_tile_center = (x, y)
                self.bench_cross_hair_position = first_tile_center

    def cycle_bench_cross_hair_left(self):
        if self.bench_cross_hair_position_index <= 0:
            self.bench_cross_hair_position_index = len(self.bench_tiles.sprites()) - 1
        else:
            self.bench_cross_hair_position_index -= 1

    def cycle_bench_cross_hair_right(self):
        if self.bench_cross_hair_position_index >= len(self.bench_tiles.sprites()) - 1:
            self.bench_cross_hair_position_index = 0
        else:
            self.bench_cross_hair_position_index += 1

    def _handle_bench_keyboard_actions(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                for tile in self.bench_tiles:
                    if self.cross_hair.collidepoint(tile.rect.center):
                        self.selected_tile = tile
            if event.key == pygame.K_z:
                if self.focus_area == "BENCH":
                    self.focus_area = "BOARD"
                else:
                    # calculate the first tile location on the bench first.
                    self.focus_area = "BENCH"
            # left arrow
            if event.key == pygame.K_LEFT:
                self.cycle_bench_cross_hair_left()
            # right arrow
            if event.key == pygame.K_RIGHT:
                self.cycle_bench_cross_hair_right()
            # up arrow
            if event.key == pygame.K_UP:
                None
            # down arrow
            if event.key == pygame.K_DOWN:
                None

    def _handle_keyboard_actions(self, event: pygame.event.Event) -> None:
        if self.focus_area == "BOARD":
            self._handle_board_keyboard_actions(event)
        else:
            self._handle_bench_keyboard_actions(event)

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
                tile.change_background_color(self.model.board_valid, self.model.victory)
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

        if self.focus_area == "BOARD":
            self.cross_hair = GameRenderer.draw_cross_hair(
                self.cross_hair_position, 30, 30
            )
            pygame.draw.rect(self.screen, "red", self.cross_hair, 2)
        else:
            self.init_bench_cross_hair()
            self.cross_hair = GameRenderer.draw_cross_hair(
                self.bench_cross_hair_position,
                20,
                20,
            )
            pygame.draw.rect(self.screen, "red", self.cross_hair, 2)

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
