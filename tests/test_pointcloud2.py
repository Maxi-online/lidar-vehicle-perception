import struct
from types import SimpleNamespace

import numpy as np
import pytest

from lidar_vehicle_perception.pointcloud2 import decode_pointcloud2


def message(big_endian=False):
    data = bytearray(2 * 40)  # Two padded rows, two XYZI points per row.
    expected = []
    for row in range(2):
        for col in range(2):
            xyz = [row * 2 + col, 2.0, 3.0]
            struct.pack_into(
                (">" if big_endian else "<") + "ffff", data, row * 40 + col * 16, *xyz, 0.5
            )
            expected.append(xyz)
    fields = [
        SimpleNamespace(name=name, datatype=7, count=1, offset=offset)
        for name, offset in [("x", 0), ("y", 4), ("z", 8), ("intensity", 12)]
    ]
    return SimpleNamespace(
        data=data,
        fields=fields,
        width=2,
        height=2,
        point_step=16,
        row_step=40,
        is_bigendian=big_endian,
    ), expected


@pytest.mark.parametrize("big_endian", [False, True])
def test_xyzi_padding_and_byte_order(big_endian):
    cloud, expected = message(big_endian)
    np.testing.assert_allclose(decode_pointcloud2(cloud), expected)


def test_truncated_buffer_is_rejected():
    cloud, _ = message()
    cloud.data = cloud.data[:-1]
    with pytest.raises(ValueError, match="truncated"):
        decode_pointcloud2(cloud)


def test_missing_xyz_field_is_rejected():
    cloud, _ = message()
    cloud.fields = cloud.fields[1:]
    with pytest.raises(ValueError, match="requires"):
        decode_pointcloud2(cloud)
