"""Decode XYZ using PointCloud2 fields, stride and endianness; no ROS import needed."""

import numpy as np


def decode_pointcloud2(message):
    fields = {field.name: field for field in message.fields}
    if not all(name in fields for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 requires x, y and z fields")
    width, height = int(message.width), int(message.height)
    point_step, row_step = int(message.point_step), int(message.row_step)
    if min(width, height) < 0 or point_step <= 0 or row_step < width * point_step:
        raise ValueError("invalid PointCloud2 dimensions or strides")
    if len(message.data) < row_step * height:
        raise ValueError("truncated PointCloud2 buffer")
    if width == 0 or height == 0:
        return np.empty((0, 3))
    order = ">" if message.is_bigendian else "<"
    columns = []
    for name in ("x", "y", "z"):
        field = fields[name]
        if field.datatype not in (7, 8) or field.count != 1:
            raise ValueError("XYZ fields must be scalar FLOAT32 or FLOAT64")
        dtype = np.dtype(order + ("f4" if field.datatype == 7 else "f8"))
        if field.offset < 0 or field.offset + dtype.itemsize > point_step:
            raise ValueError("XYZ field offset exceeds point stride")
        column = np.ndarray(
            (height, width),
            dtype=dtype,
            buffer=message.data,
            offset=field.offset,
            strides=(row_step, point_step),
        )
        columns.append(column.reshape(-1))
    points = np.column_stack(columns).astype(np.float64)
    return points[np.isfinite(points).all(axis=1)]
