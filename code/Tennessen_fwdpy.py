"""
This uses fwdpy11 >= 0.6.0
"""

import fwdpy11
import numpy as np
import time
import sys
import moments
import pickle
import argparse
import os

# import demography
# import networkx as nx

from scipy.sparse import coo_matrix, csr_matrix

assert fwdpy11.__version__ >= '0.6.0', "Require fwdpy11 v. 0.6.0 or higher"


def make_parser():
    parser = argparse.ArgumentParser("IM.py",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, help="Random number seed")

    optional = parser.add_argument_group("Optional")
    # demographic parameters, all given in physical units
    optional.add_argument('--Nref', type=int, default=7310,
                          help="Ancestral population size")
    optional.add_argument('--NAfr0', type=int, default=14474,
                          help="Initial size change")
    optional.add_argument('--NB', type=int, default=1861,
                          help="Eurasian bottleneck size")
    optional.add_argument('--NAfr', type=int, default=424000,
                          help="Final size of Afr population")
    optional.add_argument('--NEur0', type=int, default=1032,
                          help="Eurasian second bottleneck size")
    optional.add_argument('--NEur1', type=int, default=9237,
                          help="Eurasian size just as rapid growth starts")
    optional.add_argument('--NEur', type=int, default=512000,
                          help="Final size of Eur population")
    optional.add_argument('--T_Af', type=int, default=148000)
    optional.add_argument('--T_B', type=int, default=51000)
    optional.add_argument('--T_Eu_As', type=int, default=23000)
    optional.add_argument('--T_accel', type=int, default=5115)
    optional.add_argument('--mB', type=float,
                          default=15e-5,
                          help="Migration rate during Eurasian bottleneck")
    optional.add_argument('--mF', type=float,
                          default=2.5e-5,
                          help="Migration rate during expansion phase")
    optional.add_argument('--generation_time', '-g', default=25, type=int)

    # genome options
    optional.add_argument('--selection_coeff', default=0.0,
                          type=float, help="s from 2Ns")
    # Require 0 <= h <= 2
    optional.add_argument('-H', type=float, default=1.0, help="Dominance")
    optional.add_argument('--mutation_rate', '-u', default=1.25e-8, type=float)
    optional.add_argument('--length', '-L', default=100000, type=float)
    optional.add_argument('--recombination_rate', '-r',
                          default=2e-8, type=float)

    # sampling optinos
    optional.add_argument('--nreps', type=int, default=1,
                          help="Number of forward simulation replicates")
    optional.add_argument('--nsam', type=int, default=15,
                          help="Number of diploids to sample from each deme")
    return parser


# Functions to run simulation and track sizes and events (following tutorial):
def setup_and_run_model(pop, ddemog, simlen, simplification_time=0,
                        recorder=None, seed=13, R=0):
    """
    simlen: total simulation time
    genome length is 1, and R is the total recombination distance across this length
    simplification_time is set if we want to finish the last number of generations
        with shorter simplification intervals
    """
    pdict = {'nregions': [],
             'sregions': [],
             # set the recombintion rate to R, where R = rho/4N = r*L
             'recregions': [fwdpy11.PoissonInterval(0, 1, R)],
             # set mutation rates to zero
             'rates': (0, 0, None),
             'prune_selected': True,
             'gvalue': fwdpy11.Multiplicative(2.),
             'demography': ddemog,
             'simlen': simlen-simplification_time
             }
    params = fwdpy11.ModelParams(**pdict)
    rng = fwdpy11.GSLrng(seed)
    fwdpy11.evolvets(rng, pop, params, 100, recorder)
    if simplification_time > 0:
        md = np.array(pop.diploid_metadata, copy=False)
        print(np.unique(md['deme'], return_counts=True))
        # finish simulation with shorter simplification time
        # do we need to reset rng?
        pdict['simlen'] = simplification_time
        params = fwdpy11.ModelParams(**pdict)
        fwdpy11.evolvets(rng, pop, params, 10, recorder)


class SizeTracker(object):
    def __init__(self):
        self.data = []

    def __call__(self, pop, sampler):
        md = np.array(pop.diploid_metadata, copy=False)
        self.data.append((pop.generation, pop.N,
                          np.unique(md['deme'], return_counts=True)))


def build_discrete_demography(args):
    # List of demographic events:
    # keep track of size change, copying, and migration rate change events in
    # separate lists
    size_change = []
    copy = []
    mig_rates = []
    growth_rates = []

    # number of generations in epochs
    T0 = np.rint((args.T_Af - args.T_B) /
                 args.generation_time).astype(int)  # pre-split
    # split to bottleneck, no growth
    T1 = np.rint((args.T_B - args.T_Eu_As)/args.generation_time).astype(int)
    T2 = np.rint((args.T_Eu_As - args.T_accel) /
                 args.generation_time).astype(int)  # Eu growth with r_Eu0
    # accelerated growth in Af and Eu
    T3 = np.rint(args.T_accel/args.generation_time).astype(int)

    M_init = np.zeros(4).reshape(2, 2)
    M_init[0, 0] = 1
    mm = fwdpy11.MigrationMatrix(M_init)

    # burn in for 20*Ne generations
    gens_burn_in = 20*args.Nref
    total_sim_length = gens_burn_in+T0+T1+T2+T3

    # init: size change of common ancestral population
    size_change.append(fwdpy11.SetDemeSize(
        when=gens_burn_in, deme=0, new_size=args.NAfr0))

    # T0: mass migration, copy from A to Eu bottleneck population
    copy.append(fwdpy11.copy_individuals(when=gens_burn_in+T0, source=0,
                                         destination=1, fraction=args.NB/args.NAfr0))
    size_change.append(fwdpy11.SetDemeSize(
        when=gens_burn_in+T0, deme=1, new_size=args.NB))
    # at the same time, set migration rate between deme 0 and 1 to m_A_B
    mig_rates.append(fwdpy11.SetMigrationRates(
        gens_burn_in+T0, 0, [1-args.mB, args.mB]))
    mig_rates.append(fwdpy11.SetMigrationRates(
        gens_burn_in+T0, 1, [args.mB, 1-args.mB]))

    # T1: adjust size of Eu to Eu0 and set growth rate
    size_change.append(fwdpy11.SetDemeSize(when=gens_burn_in+T0+T1, deme=1,
                                           new_size=args.NEur0))
    r_Eur0 = (args.NEur1/args.NEur0)**(1/T2) - 1
    growth_rates.append(fwdpy11.SetExponentialGrowth(when=gens_burn_in+T0+T1,
                                                     deme=1, G=1+r_Eur0))
    # set migration rates to contemporary rates
    mig_rates.append(fwdpy11.SetMigrationRates(gens_burn_in+T0+T1, 0,
                                               [1-args.mF, args.mF]))
    mig_rates.append(fwdpy11.SetMigrationRates(gens_burn_in+T0+T1, 1,
                                               [args.mF, 1-args.mF]))

    # T2: set growth rates to accelerated rates in both populations
    r_AfrF = (args.NAfr/args.NAfr0)**(1/T3) - 1
    r_EurF = (args.NEur/args.NEur1)**(1/T3) - 1
    growth_rates.append(fwdpy11.SetExponentialGrowth(when=gens_burn_in+T0+T1+T2,
                                                     deme=0, G=1+r_AfrF))
    growth_rates.append(fwdpy11.SetExponentialGrowth(when=gens_burn_in+T0+T1+T2,
                                                     deme=1, G=1+r_EurF))

    ddemog = fwdpy11.DiscreteDemography(mass_migrations=copy, set_deme_sizes=size_change,
                                        migmatrix=mm, set_migration_rates=mig_rates,
                                        set_growth_rates=growth_rates)
    return ddemog, total_sim_length


def per_deme_sfs(pop):
    """
    Get the marginal SFS per deme.
    """
    samples = pop.alive_nodes
    md = np.array(pop.diploid_metadata, copy=False)
    deme_sizes = np.unique(md['deme'], return_counts=True)
    deme_sfs = {}
    for i, j in zip(deme_sizes[0], deme_sizes[1]):
        deme_sfs[i] = np.zeros(2*j + 1)

    ti = fwdpy11.TreeIterator(pop.tables, samples, update_samples=True)
    nt = np.array(pop.tables.nodes, copy=False)
    nmuts = 0
    # nmuts_sites = []

    row = []
    col = []
    data = []

    for tree in ti:
        for mut in tree.mutations():
            # nmuts_sites.append(pop.tables.sites[mut.site].position)
            sb = tree.samples_below(mut.node)
            dc = np.unique(nt['deme'][sb], return_counts=True)
            assert dc[1].sum() == len(sb), f"{dc[1].sum} {len(sb)}"
            nmuts += 1
            for deme, daf in zip(dc[0], dc[1]):
                deme_sfs[deme][daf] += 1
            if 0 not in dc[0]:
                deme_sfs[0][0] += 1
            if 1 not in dc[0]:
                deme_sfs[1][0] += 1
            data.append(1)
            if 0 in dc[0]:
                row.append(dc[1][0])
                if 1 in dc[0]:
                    col.append(dc[1][1])
                else:
                    col.append(0)
            else:
                row.append(0)
                col.append(dc[1][0])

    jSFS_coo = coo_matrix((data, (row, col)),
                          shape=(len(deme_sfs[0]), len(deme_sfs[1])))
    jSFS = csr_matrix(jSFS_coo)

    return moments.Spectrum(deme_sfs[0]), moments.Spectrum(deme_sfs[1]), jSFS


def project_sfs(sfs, n):
    fs = moments.Spectrum(sfs)
    psfs = fs.project([n])
    return psfs.data[1:-1]


def tennessen_moments(args):
    # ns = diploid size
    sample_sizes = (2*args.nsam, 2*args.nsam)
    fs = moments.Demographics1D.snm([sum(sample_sizes)])
    fs.integrate([args.NAfr0/args.Nref], (args.T_Af-args.T_B) /
                 args.generation_time/2/args.Nref)
    fs = moments.Manips.split_1D_to_2D(fs, sample_sizes[0], sample_sizes[1])
    fs.integrate([args.NAfr0/args.Nref, args.NB/args.Nref],
                 (args.T_B-args.T_Eu_As)/args.generation_time/2/args.Nref,
                 m=[[0, 2*args.Nref*args.mB], [2*args.Nref*args.mB, 0]])
    def nu_func(t): return [args.NAfr0/args.Nref,
                            args.NEur0/args.Nref * np.exp(np.log(args.NEur1/args.NEur0) * t / ((args.T_Eu_As-args.T_accel)/args.generation_time/2/args.Nref))]
    fs.integrate(nu_func, (args.T_Eu_As-args.T_accel)/args.generation_time/2/args.Nref,
                 m=[[0, 2*args.Nref*args.mF], [2*args.Nref*args.mF, 0]])
    def nu_func(t): return [args.NAfr0/args.Nref * np.exp(np.log(args.NAfr/args.NAfr0) * t / (args.T_accel/args.generation_time/2/args.Nref)),
                            args.NEur1/args.Nref * np.exp(np.log(args.NEur/args.NEur1) * t / (args.T_accel/args.generation_time/2/args.Nref))]
    fs.integrate(nu_func, args.T_accel/args.generation_time/2/args.Nref,
                 m=[[0, 2*args.Nref*args.mF], [2*args.Nref*args.mF, 0]])
    return fs


# def tennessen_demography(args):
#     G = nx.DiGraph()
#     # convert to genetic units
#     T0 = (args.T_Af-args.T_B)/args.generation_time/2/args.Nref
#     T1 = (args.T_B-args.T_Eu_As)/args.generation_time/2/args.Nref
#     T2 = (args.T_Eu_As-args.T_accel)/args.generation_time/2/args.Nref
#     T3 = (args.T_accel)/args.generation_time/2/args.Nref
#     MB = 2*args.Nref*args.mB
#     MF = 2*args.Nref*args.mF
#     nuA = args.NAfr0/args.Nref
#     nuB = args.NB/args.Nref
#     nuAfr = args.NAfr/args.Nref
#     nuEur = args.NEur/args.Nref
#     nuEur0 = args.NEur0/args.Nref
#     nuEur1 = args.NEur1/args.Nref
#     # add edges and nodes
#     G.add_node('root', nu=1, T=0)
#     G.add_node('A', nu=nuA, T=T0)
#     G.add_node('B', nu=nuB, T=T1, m={'Afr0': MB})
#     G.add_node('Afr0', nu=nuA, T=T1+T2, m={'B': MB, 'Eur1': MF})
#     G.add_node('Eur1', nu0=nuEur0, nuF=nuEur1, T=T2, m={'Afr0': MF})
#     G.add_node('Eur', nu0=nuEur1, nuF=nuEur, T=T3, m={'Afr': MF})
#     G.add_node('Afr', nu0=nuA, nuF=nuAfr, T=T3, m={'Eur': MF})
#     G.add_edges_from([('root','A'), ('A','B'), ('A','Afr0'), ('Afr0','Afr'), ('B','Eur1'), ('Eur1','Eur')])
#     dg = demography.DemoGraph(G)
#     fs = dg.SFS(pop_ids=['Afr','Eur'], sample_sizes=[2*args.nsam, 2*args.nsam])
#     return fs

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    # set up random seeds
    np.random.seed(args.seed)
    seeds = np.random.randint(0, np.iinfo(np.uint32).max, args.nreps)

    for rep in range(args.nreps):
        print(f"running rep {rep+1} of {args.nreps}")
        rng = fwdpy11.GSLrng(seeds[rep])

        # Initialize demes:
        # Deme 0: Ancestral and Afr, Deme 1: Eur
        pop = fwdpy11.DiploidPopulation([args.Nref, 0], 1.0)
        ddemog, total_sim_length = build_discrete_demography(args)
        freq_simplification_time = np.rint(
            args.T_accel/args.generation_time).astype(int)

        # set up total recombination rate for sim
        r = args.recombination_rate
        L = args.length
        R = r*L

        st = SizeTracker()

        time1 = time.time()
        print(freq_simplification_time, total_sim_length)
        setup_and_run_model(pop, ddemog, total_sim_length, simplification_time=freq_simplification_time,
                            R=R, seed=seeds[rep])
        time2 = time.time()

        # print(f"length of simulation: {L}")
        print(f"time to run simulation: {time2-time1}")
        md = np.array(pop.diploid_metadata, copy=False)
        print(np.unique(md['deme'], return_counts=True), [args.NAfr, args.NEur])
        # assert np.all(st.data[-1][2][1] == [args.NAfr,args.NEur]), "final sizes aren't right"

        # add neutral mutations
        fwdpy11.infinite_sites(rng, pop, args.length * args.mutation_rate)

        # get full frequency spectra
        fs0, fs1, jSFS = per_deme_sfs(pop)

        # project to desired sizes
        fs0_proj = fs0.project([2*args.nsam])
        fs1_proj = fs1.project([2*args.nsam])

        # moments spectrum
        theta = 4 * args.Nref * args.mutation_rate * args.length
        F = tennessen_moments(args) * theta
        F0 = F.marginalize([1])
        F1 = F.marginalize([0])

        spectra = {'fwdpy': {'Afr': fs0_proj, 'Eur': fs1_proj},
                   'moments': {'Afr': F0, 'Eur': F1}}

        # fname = f'/lb/project/gravel/ragsdale_projects/PRS/simulations/spectra_tennessen_ns_{args.nsam}_length_{args.length}_seed_{seeds[rep]}.bp'
        # with open(fname, 'wb+') as fout:
        #     pickle.dump(spectra, fout)
