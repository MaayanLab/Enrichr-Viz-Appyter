# Enrichr-Library-Annealing

This repository serves as a storage system for annealed libraries from Enrichr which are used in the Enrichr-Canvas-Appyter (available here: https://appyters.maayanlab.cloud/Enrichr_Canvas_Appyter/)

library_processor.ipynb features the code used to anneal the libraries. The library is opened from Enrichr and each gene set in the library becomes one hexagon on a continuous hexagonal canvas. Then, the fitness of the canvas is found by computing the jaccard similarity of each hexagon and it's immediate neighbors and summing all of the indices. The algorithm then chooses 2 random hexagons to "swap", recalculates the fitness of the system, and keeps the swap if the fitness increases.

The annealed libraries are stored in a list of tuples of format (name, gene_list) in binary encoding using python's pickle package.