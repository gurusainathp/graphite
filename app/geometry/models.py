"""
Graphite — models.py
Responsibility: Clean internal geometry data structures.
No raw OpenCV arrays beyond this point.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Point:
    """A 2D point in either image or normalized coordinate space."""
    x: float
    y: float

    def __repr__(self) -> str:
        return f"Point({self.x:.2f}, {self.y:.2f})"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box around a contour."""
    x: float        # top-left x
    y: float        # top-left y
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x:.1f}, y={self.y:.1f}, w={self.width:.1f}, h={self.height:.1f})"


@dataclass
class Contour:
    """
    A single extracted contour with its geometry properties.

    Attributes:
        points:       Ordered list of (x, y) points along the contour boundary.
        area:         Pixel area enclosed by the contour.
        perimeter:    Arc length of the contour.
        bbox:         Axis-aligned bounding box.
        is_external:  True if this is an outer contour (no parent in hierarchy).
        hierarchy_id: Index from cv2.findContours hierarchy array.
    """
    points:       List[Point]
    area:         float
    perimeter:    float
    bbox:         BoundingBox
    is_external:  bool  = True
    hierarchy_id: int   = -1

    @property
    def point_count(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        kind = "external" if self.is_external else "internal"
        return (
            f"Contour({kind}, points={self.point_count}, "
            f"area={self.area:.1f}, perimeter={self.perimeter:.1f})"
        )


@dataclass
class ContourCollection:
    """All contours extracted from a single edge image."""
    contours:      List[Contour] = field(default_factory=list)
    image_width:   int = 0
    image_height:  int = 0

    @property
    def count(self) -> int:
        return len(self.contours)

    @property
    def external(self) -> List[Contour]:
        return [c for c in self.contours if c.is_external]

    @property
    def internal(self) -> List[Contour]:
        return [c for c in self.contours if not c.is_external]

    def print_stats(self) -> None:
        areas = [c.area for c in self.contours]
        print(f"[models] Contours     : {self.count}")
        print(f"[models] External     : {len(self.external)}")
        print(f"[models] Internal     : {len(self.internal)}")
        if areas:
            print(f"[models] Area range   : {min(areas):.1f} – {max(areas):.1f} px²")
            print(f"[models] Avg area     : {sum(areas)/len(areas):.1f} px²")