## Simple Result

The largest numerical value in the document is: 675329.0
Found on page: 92

## Using

The code does use flash attention 2 which requires either Ampere, Ada, or Hopper GPU archs.
Assuming you already have the correct version of torch with CUDA enabled.

```
pip install requirements.txt
pip install qwen-vl-utils[decord]
pip install flash-attn --no-build-isolation
```

### Final Results:

The resulting numbers below were run on a single H100 with basically no optimization.

```
NOTE:
The result is incorrect, but was seen to be corrected through better prompting but resulted in slower processing.
```

Total numbers found: 89
Maximum value found: 96,496,000,000 (on page 82)

Top 5 Largest Numbers:

- 96,496,000,000 (on page 82)
- 32,119,000,000 (on page 40)
- 23,497,000,000 (on page 84)
- 21,091,000,000 (on page 86)
- 19,977,000,000 (on page 85)

Timing Statistics:
Total execution time: 32.10 seconds
Total processing time: 32.09 seconds
Average time per page: 0.28 seconds

