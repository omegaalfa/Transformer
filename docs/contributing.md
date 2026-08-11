# Contributing

Do not implement optimization before numerical correctness and parity tests
exist. The required sequence is correctness, parity, profiling, then
optimization.

Use strict types, PSR-12, precise PHPDoc for structured arrays, and typed DTOs.
Keep high-level code isolated from FFI, Zend, native pointers, and devices.
Every mathematical implementation must arrive with focused unit and parity
tests; do not add tests that pass without asserting real behavior.
