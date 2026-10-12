"""Label normalization and building-floor map loading."""

from dataclasses import dataclass
import re
from pathlib import Path

import yaml


class FloorMapError(ValueError):
    """Base class for floor-map validation and lookup errors."""


class MissingBuildingError(FloorMapError):
    """Raised when a requested building is absent."""


class MissingFloorError(FloorMapError):
    """Raised when a requested floor is absent."""


@dataclass(frozen=True)
class FloorEntry:
    displayed_label: str
    aliases: tuple


def normalize_label(value):
    if value is None:
        return ''
    label = ' '.join(str(value).strip().upper().replace('_', ' ').split())
    if not label:
        return ''
    if label in {'G', 'GF', 'GROUND'}:
        return 'G'
    if label in {'PH', 'PENTHOUSE'}:
        return 'PH'
    if label in {'DOOR OPEN', 'OPEN DOOR'}:
        return 'DOOR_OPEN'
    if label in {'DOOR CLOSE', 'CLOSE DOOR'}:
        return 'DOOR_CLOSE'
    basement = re.fullmatch(r'B-?0*(\d+)', label)
    if basement:
        return 'B{}'.format(int(basement.group(1)))
    suffix_b = re.fullmatch(r'0*(\d+)B', label)
    if suffix_b:
        return '{}B'.format(int(suffix_b.group(1)))
    floor_number = re.fullmatch(r'FLOOR\s+0*(\d+)', label)
    if floor_number:
        return str(int(floor_number.group(1)))
    number = re.fullmatch(r'0*(\d+)', label)
    if number:
        return str(int(number.group(1)))
    return label


class FloorMap:
    def __init__(self, buildings):
        if not isinstance(buildings, dict):
            raise FloorMapError('buildings must be a mapping')
        self._buildings = buildings

    @classmethod
    def load(cls, path):
        map_path = Path(path)
        if not map_path.is_file():
            raise FloorMapError('floor-map file does not exist: {}'.format(map_path))
        with map_path.open('r', encoding='utf-8') as stream:
            payload = yaml.safe_load(stream)
        if not isinstance(payload, dict) or not isinstance(payload.get('buildings'), dict):
            raise FloorMapError("floor-map root must contain a 'buildings' mapping")
        return cls(payload['buildings'])

    def get_floor(self, building_id, requested_floor):
        building = self._buildings.get(building_id)
        if not isinstance(building, dict):
            raise MissingBuildingError('missing building mapping: {}'.format(building_id))
        floors = building.get('floors')
        if not isinstance(floors, dict):
            raise FloorMapError('building floors must be a mapping: {}'.format(building_id))
        floor = floors.get(str(requested_floor))
        if not isinstance(floor, dict):
            raise MissingFloorError(
                'missing floor mapping: {}/{}'.format(building_id, requested_floor)
            )
        displayed_label = floor.get('displayed_label')
        aliases = floor.get('aliases', [])
        if not isinstance(displayed_label, str) or not displayed_label.strip():
            raise FloorMapError('displayed_label must be a non-empty string')
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise FloorMapError('aliases must be a list of strings')
        return FloorEntry(displayed_label.strip(), tuple(aliases))

    @staticmethod
    def matches(label, entry):
        allowed = {normalize_label(entry.displayed_label)}
        allowed.update(normalize_label(alias) for alias in entry.aliases)
        return normalize_label(label) in allowed
