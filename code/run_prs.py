import msprime
import numpy as np
import simulate_ooa



if __name__ == "__main__":
    mutation_rate = 2e-8
    chrom = 20
    map_path = '/Users/aragsdal/Data/Human/maps_b37/genetic_map_HapMapII_GRCh37/'
    recombination_map = map_path + f'genetic_map_GRCh37_chr{chrom}.txt'
    
    sample_sizes=[400000,400000,400000]
    ts = simulate_ooa.get_tree_sequeneces(recombination_map=recombination_map,
                                          sample_sizes=sample_sizes,
                                          mutation_rate=mutation_rate)


