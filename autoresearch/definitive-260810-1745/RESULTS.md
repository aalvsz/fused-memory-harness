# Frozen definitive result

The benchmark contains 1,100 synthetic cases, 14 independently executed memory
methods, and four context budgets for 61,600 method--case--budget rows.

At the primary 8,000-character budget, Fused Memory retrieved 1,100/1,100
targets. Gated RRF retrieved 1,008/1,100; dense + recent and union + recent each
retrieved 1,003/1,100; the legacy hybrid retrieved 998/1,100.

The primary and reproduction runs produced the same deterministic decision hash:

`5efde9fbedbd4e4125556a98a84e19ea8e4356f53de49cb5d05808a58fd71688`

The dataset hash is:

`9aff1accbbee8e6aeb4aaba2c509ac44d2a4199b30703b17b91a9faca261e474`

The raw primary and reproduction context JSONL files are each approximately
134 MiB and are intentionally omitted from normal Git tracking. They can be
regenerated using the commands in the repository README. Compact summaries,
pairwise comparisons, validation output, the frozen configuration, and the
synthetic dataset are retained here.
