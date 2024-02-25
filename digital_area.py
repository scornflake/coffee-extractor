class Area:
    left: int
    top: int
    width: int
    height: int

    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def top_left(self) -> tuple:
        return (self.left, self.top,)

    @property
    def top_right(self) -> tuple:
        return (self.right, self.top,)

    @property
    def bottom_left(self) -> tuple:
        return (self.left, self.bottom,)

    def bottom_left_skewed(self, skew) -> tuple:
        return (self.left + skew, self.bottom,)

    def bottom_right_skewed(self, skew) -> tuple:
        return (self.right + skew, self.top + self.height,)

    @property
    def bottom_right(self) -> tuple:
        return (self.right, self.top + self.height,)
