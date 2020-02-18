"""
Run from same dir as simulate_ooa.py and effect_sizes.py
"""
import msprime
import numpy as np
import simulate_ooa
import effect_sizes
import sys
import argparse
import os

# To do: fully document code and check for other bugs

from os.path import expanduser
home = expanduser("~")
# the genetic maps are in ~/Data/Human/maps_b37/genetic_map_HapMapII_GRCh37


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main(args):
    eprint(
        f'Starting simulations: h2={args.h2}, ncausal={args.ncausal}, alpha={args.alpha}')
    mutation_rate = 2e-8
    chrom = args.chrom

    map_path = home+'/Data/Human/maps_b37/genetic_map_HapMapII_GRCh37/'
    recombination_map = map_path + f'genetic_map_GRCh37_chr{chrom}.txt'
    if os.path.isfile(recombination_map) is False:
        map_path = '/lb/project/gravel/data/GeneticMap_HapMapII/'
        recombination_map = map_path + f'genetic_map_GRCh37_chr{chrom}.txt'

    nhaps = list(map(int, args.nhaps.split(',')))

    # generate/load coalescent simulations
    tree_path = os.path.join(args.path,
                             args.out+'_nhaps_'
                             + '_'.join(args.nhaps.split(','))
                             + '.hdf5')
    try:
        eprint("Trying to load simulated trees")
        ts = msprime.load(tree_path)
    except:
        eprint("Trees not found, simulating")
        ts = simulate_ooa.get_tree_sequeneces(
            recombination_map=recombination_map,
            sample_sizes=nhaps,
            mutation_rate=mutation_rate)
        ts.dump(tree_path)

    eprint('Simulation:')
    eprint('Number of haplotypes: ' + ','.join(map(str, nhaps)))
    eprint('Number of trees: ' + str(ts.get_num_trees()))
    eprint('Number of mutations: ' + str(ts.get_num_mutations()))
    eprint('Sequence length: ' + str(ts.get_sequence_length()))

    # compute effect sizes for causal snps
    prs_true = effect_sizes.true_prs(ts, args.ncausal, args.h2, nhaps,
                                     os.path.join(args.path, args.out), args.alpha)
    # get list of cases and controls in european samples
    cases_diploid, controls_diploid, prs_norm, environment = effect_sizes.case_control(
        prs_true, args.h2, nhaps, args.prevalence, args.ncontrols)
    # run the gwas on these cases/controls
    summary_stats, cases_haploid, controls_haploid = effect_sizes.run_gwas(
        ts, cases_diploid, controls_diploid, args.p_threshold, args.cc_maf)
    #
    clumped_snps, usable_positions = effect_sizes.clump_variants(ts,
                                                                 summary_stats, nhaps, args.r2, args.window_size)
    #
    prs_infer = effect_sizes.infer_prs(ts, nhaps, clumped_snps,
                                       summary_stats, usable_positions, args.h2, args.ncausal,
                                       os.path.join(args.path, args.out), args.alpha)
    #
    effect_sizes.write_summaries(os.path.join(args.path, args.out),
                                 prs_true, prs_infer, nhaps, cases_diploid,
                                 controls_diploid, args.h2, args.ncausal, environment, args.alpha)

    eprint('Ending simulations')
    eprint('\n')


if __name__ == "__main__":
    # set path to send simulation and intermidiates
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhaps', help='AFR,EUR,EAS',
                        default='400000,400000,400000')
    parser.add_argument('--chrom', help='', default='20')
    parser.add_argument('--ncausal', help='', type=int, default=200)
    parser.add_argument('--ncontrols', help='', type=int, default=10000)
    parser.add_argument('--h2', help='', type=float, default=float(2)/3)
    parser.add_argument('--prevalence', help='', type=float, default=0.05)
    parser.add_argument('--p_threshold', help='', type=float, default=0.01)
    parser.add_argument('--cc_maf', help='', type=float, default=0.01)
    parser.add_argument('--r2', help='', type=float, default=0.5)
    parser.add_argument('--window_size', help='', type=int, default=250e3)
    parser.add_argument('--out', help='', default='sim0')
    parser.add_argument('--path', help='', default=home +
                        '/Project/PRS/simulations/')
    parser.add_argument('--alpha', help='', type=float, default='-1')

    args = parser.parse_args()
    main(args)
