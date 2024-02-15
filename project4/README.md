# Project 4

## Part 1

### Description

This part uses **K-means clustering** for image compression. The input images are [Koala.jpg](./part1/Koala.jpg) and [Penguins.jpg](./part1/Penguins.jpg) and the compressed output images are [here](./part1/tests/) with varying values of **k** and **p**(order of Minkowski distance used to calculate the distance between 2 pixels).

## Compile and Run

Run Tests: `bash run.sh` or with `./run.sh`, but you may need to change permissions of the shell script with `chmod +x run.sh`.
Must have Java JDK >= 11 to compile the code.

This will create 'tests' directory and run tests on both the provided input images.

Output images are named `tests/<input-image>-<k>-<p>.jpg` where k is clusters for k-means algorithm and p is the order of the Minkowski distance.

## Part 2

### Description

In this part, I used the [Chow-Liu trees](https://jmlr.csail.mit.edu/papers/volume1/meila00a/meila00a.pdf) to implement **Tree Bayesian networks**, **Mixtures of Tree Bayesian networks using Expecation maximization** and **Mixtures of Tree Bayesian networks using Random Forests** for a variety of datasets. The datasets are NLTCS, MSNBC, KDDCup 2000, Plants, Audio, Jester, Netflix, Accidents, Retail, Pumsb-star. 

## Reports

- [Part1 - Report](./Part1-Report.pdf)
- [Part2 - Report](./Part2-Report.pdf)
