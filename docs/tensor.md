# Tensor architecture

`Tensor` is the central value exposed by the PHP API. It owns a `Shape` and a
`StorageInterface`, so execution is not permanently coupled to PHP arrays.
Planned storage implementations are educational PHP-array storage, opaque
native storage, and future GPU storage.

The public surface reserves construction, metadata, indexing, element-wise,
reduction, algebra, and activation methods. Every operation is currently a
non-functional contract. Shape validation, strides, broadcasting, dtype
conversion, allocation, and mathematics belong to later milestones.

Materialization as a PHP array will eventually be an explicit boundary such as
`toArray()`; it must never happen implicitly between native operations.
