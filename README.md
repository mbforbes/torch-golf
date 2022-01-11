# torch-golf

🔥⛳️

## linear regression

timing:
- data generation (13.5m)
- plotting data (4.5m)
- regression (1h 14m)
    - doing derivation by hand then remembering I don't have to cause autograd
    - starting to write code then copilot spitting out the rest
    - debugging why loss was exploding
    - failing that, Googling basic examples
    - OMG the learning rate was just too high. FML.
        - why does it explode and not even go in the right direction?
        - can we graph the loss landscape given it's a 1D problem?
        - cheat sheet for, e.g., normalization?

## loss landscape
