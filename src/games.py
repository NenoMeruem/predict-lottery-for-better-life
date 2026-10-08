"""Cấu hình các loại vé dạng "chọn k số trong N" dùng chung cho cleaner/analyzer/visualizer/predictor."""
from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class GameSpec:
    code: str
    name: str
    n_main: int
    max_number: int
    has_bonus: bool = False
    # Số bonus nằm trong khoảng 1..bonus_max (mặc định = max_number)
    bonus_max: int | None = None
    # True: bonus quay từ lồng riêng (có thể trùng số chính, VD Lotto 5/35).
    # False: bonus rút tiếp từ cùng lồng số chính (không trùng, VD Power 6/55).
    bonus_separate: bool = False
    # True: người chơi tự chọn số bonus khi mua vé (Lotto 5/35)
    bonus_player_picks: bool = False

    @property
    def main_cols(self) -> list[str]:
        return [f"n{i + 1}" for i in range(self.n_main)]

    @property
    def numbers(self) -> range:
        return range(1, self.max_number + 1)

    @property
    def bonus_range_max(self) -> int:
        return self.bonus_max or self.max_number

    @property
    def bonus_numbers(self) -> range:
        return range(1, self.bonus_range_max + 1)

    @property
    def p_number(self) -> float:
        """Xác suất một số cụ thể xuất hiện trong bộ số chính của một kỳ."""
        return self.n_main / self.max_number

    @property
    def p_pair(self) -> float:
        """Xác suất một cặp số cụ thể cùng xuất hiện trong một kỳ."""
        return comb(self.n_main, 2) / comb(self.max_number, 2)


GAMES: dict[str, GameSpec] = {
    "535": GameSpec("535", "Lotto 5/35", n_main=5, max_number=35, has_bonus=True,
                    bonus_max=12, bonus_separate=True, bonus_player_picks=True),
    "645": GameSpec("645", "Mega 6/45", n_main=6, max_number=45, has_bonus=False),
    "655": GameSpec("655", "Power 6/55", n_main=6, max_number=55, has_bonus=True),
}


def get_spec(game: str) -> GameSpec:
    try:
        return GAMES[str(game)]
    except KeyError:
        raise ValueError(f"Game không hỗ trợ: {game!r}. Chọn một trong {list(GAMES)}") from None
