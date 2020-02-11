"""
This uses fwdpy11 >= 0.6.0
"""

import fwdpy11
import numpy as np
import time
import sys
import moments
import argparse
import pickle

from scipy.sparse import coo_matrix, csr_matrix

assert fwdpy11.__version__ >= '0.6.0', "Require fwdpy11 v. 0.6.0 or higher"

def make_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--length', '-l', type=float, default=100e3,
                        help="length of genome")
    parser.add_argument('--mutation_rate', '-u', type=float, default=1.25e-8,
                        help="mutation rate")
    parser.add_argument('--recombination_rate', '-r', type=float, default=2e-8,
                        help="recombination rate")
    parser.add_argument('--seed', type=int, default=13,
                        help="random seed")
    parser.add_argument('--nreps', type=int, default=1,
                        help="Number of forward simulation replicates")
    parser.add_argument('--nsam', type=int, default=15,
                        help="Number of diploids to sample from each deme")
    return parser

def set_demographic_parameters():
    # OOA parameters from Gravel et al 2011, augmented with Tennessen et al
    # raw population sizes and times in generations
    gen_time = 25

    global N_ref
    N_ref = 7310
    global N_Af0 
    N_Af0 = 14474
    global N_B
    N_B = 1861
    global N_Eu0
    N_Eu0 = 1032

    global m_Af0_B
    m_Af0_B = 0 # 15e-5
    global m_Af1_Eu1
    m_Af1_Eu1 = 0 # 2.5e-5

    T_Af = np.rint(148000 / gen_time).astype(int)
    T_B = np.rint(51000 / gen_time).astype(int)
    T_Eu_As = np.rint(23000 / gen_time).astype(int)

    global r_Eu0
    r_Eu0 = 0.307e-2

    T_accel = np.rint(5115 / gen_time).astype(int)

    # number of generations in epochs
    global T0
    T0 = T_Af - T_B # pre-split
    global T1
    T1 = T_B - T_Eu_As # split to bottleneck, no growth
    global T2
    T2 = T_Eu_As - T_accel # Eu growth with r_Eu0
    global T3
    T3 = T_accel # accelerated growth in Af and Eu

    global N_EuF
    N_EuF = 512e3
    global N_AfF
    N_AfF = 424e3

    global N_Eu1
    N_Eu1 = N_Eu0 * (1+r_Eu0) ** T2
    global r_Eu1
    r_Eu1 = (N_EuF / np.rint(N_Eu1)) ** (1./T3) - 1
    global r_Af1
    r_Af1 = (N_AfF / N_Af0) ** (1./T3) - 1

# Functions to run simulation and track sizes and events (following tutorial):
def setup_and_run_model(pop, ddemog, simlen, recorder=None, seed=13, R=0):
    """
    simlen: total simulation time
    genome length is 1, and R is the total recombination distance across this length
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
            'simlen': simlen
           }
    params = fwdpy11.ModelParams(**pdict)
    rng = fwdpy11.GSLrng(seed)
    fwdpy11.evolvets(rng, pop, params, 100, recorder)

class SizeTracker(object):
    def __init__(self):
        self.data = []
    def __call__(self, pop, sampler):
        md = np.array(pop.diploid_metadata, copy=False)
        self.data.append((pop.generation, pop.N,
                         np.unique(md['deme'], return_counts=True)))

def get_ddemog():
    # List of demographic events:
    # keep track of size change, copying, and migration rate change events in
    # separate lists
    size_change = []
    copy = []
    mig_rates = []
    growth_rates = []

    M_init = np.zeros(4).reshape(2,2)
    M_init[0,0] = 1
    mm = fwdpy11.MigrationMatrix(M_init)

    # burn in for 20*Ne generations
    gens_burn_in = 20*N_ref
    total_sim_length = gens_burn_in+T0+T1+T2+T3

    # init: size change of common ancestral population
    size_change.append(fwdpy11.SetDemeSize(when=gens_burn_in, deme=0, new_size=N_Af0))

    # T0: mass migration, copy from A to Eu bottleneck population
    copy.append(fwdpy11.copy_individuals(when=gens_burn_in+T0, source=0,
                                         destination=1, fraction=N_B/N_Af0))
    size_change.append(fwdpy11.SetDemeSize(when=gens_burn_in+T0, deme=1, new_size=N_B))
    # at the same time, set migration rate between deme 0 and 1 to m_A_B
    mig_rates.append(fwdpy11.SetMigrationRates(gens_burn_in+T0, 0, [1-m_Af0_B, m_Af0_B]))
    mig_rates.append(fwdpy11.SetMigrationRates(gens_burn_in+T0, 1, [m_Af0_B, 1-m_Af0_B]))

    # T1: adjust size of Eu to Eu0 and set growth rate
    size_change.append(fwdpy11.SetDemeSize(when=gens_burn_in+T0+T1, deme=1, 
                                           new_size=N_Eu0))
    growth_rates.append(fwdpy11.SetExponentialGrowth(when=gens_burn_in+T0+T1,
                                                   deme=1, G=1+r_Eu0))
    # set migration rates to contemporary rates
    mig_rates.append(fwdpy11.SetMigrationRates(gens_burn_in+T0+T1, 0, 
                                               [1-m_Af1_Eu1, m_Af1_Eu1]))
    mig_rates.append(fwdpy11.SetMigrationRates(gens_burn_in+T0+T1, 1, 
                                               [m_Af1_Eu1, 1-m_Af1_Eu1]))

    # T2: set growth rates to accelerated rates in both populations
    growth_rates.append(fwdpy11.SetExponentialGrowth(when=gens_burn_in+T0+T1+T2,
                                                   deme=0, G=1+r_Af1))
    growth_rates.append(fwdpy11.SetExponentialGrowth(when=gens_burn_in+T0+T1+T2,
                                                   deme=1, G=1+r_Eu1))

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
    #nmuts_sites = []
    
    row = []
    col = []
    data = []
    
    for tree in ti:
        for mut in tree.mutations():
            #nmuts_sites.append(pop.tables.sites[mut.site].position)
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
    
    #assert nmuts == len(pop.tables.mutations)
    
    #table_sites = []
    #for mut in pop.tables.mutations:
    #    table_sites.append(pop.tables.sites[mut.site].position)
    
    #assert len(pop.tables.mutations) == len(pop.tables.sites)
    
    return moments.Spectrum(deme_sfs[0]), moments.Spectrum(deme_sfs[1]), jSFS

def project_sfs(sfs, n):
    fs = moments.Spectrum(sfs)
    psfs = fs.project([n])
    return psfs.data[1:-1]

def tennessen_moments(ns):
    # ns = diploid size
    sample_sizes = (2*ns, 2*ns)
    fs = moments.Demographics1D.snm([sum(sample_sizes)])
    fs.integrate([N_Af0/N_ref], T0/2/N_ref)
    fs = moments.Manips.split_1D_to_2D(fs, 2*ns, 2*ns)
    fs.integrate([N_Af0/N_ref, N_B/N_ref], T1/2/N_ref, 
                 m=[[0,2*N_ref*m_Af0_B],[2*N_ref*m_Af0_B,0]])
    nu_func = lambda t: [N_Af0/N_ref,
                         N_Eu0/N_ref * np.exp(np.log(N_Eu1/N_Eu0) * t / (T2/2/N_ref))]
    fs.integrate(nu_func, T2/2/N_ref, 
                 m=[[0,2*N_ref*m_Af1_Eu1],[2*N_ref*m_Af1_Eu1,0]])
    nu_func = lambda t: [N_Af0/N_ref * np.exp(np.log(N_AfF/N_Af0) * t / (T3/2/N_ref)),
                         N_Eu1/N_ref * np.exp(np.log(N_EuF/N_Eu1) * t / (T3/2/N_ref))]
    fs.integrate(nu_func, T3/2/N_ref,
                 m=[[0,2*N_ref*m_Af1_Eu1],[2*N_ref*m_Af1_Eu1,0]])
    return fs

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    # set up random seeds
    np.random.seed(args.seed)
    seeds = np.random.randint(0, np.iinfo(np.uint32).max, args.nreps)

    # set the demographic parameters
    set_demographic_parameters()
    
    for rep in range(args.nreps):
        print(f"running rep {rep+1} of {args.nreps}")
        rng = fwdpy11.GSLrng(seeds[rep])
        
        # Initialize demes:
        # Deme 0: Ancestral and Afr, Deme 1: Eur
        pop = fwdpy11.DiploidPopulation([N_ref, 0], 1.0)
        ddemog, total_sim_length = get_ddemog()

        # set up total recombination rate for sim
        r = args.recombination_rate
        L = args.length
        R = r*L

        st = SizeTracker()

        time1 = time.time()
        setup_and_run_model(pop, ddemog, total_sim_length, recorder=st, R=R, seed=seeds[rep])
        time2 = time.time()

        print(f"length of simulation: {L}")
        print(f"time to run simulation: {time2-time1}")
        assert np.all(st.data[-1][2][1] == [424000,512000]), "final sizes aren't right"

        # add neutral mutations
        fwdpy11.infinite_sites(rng, pop, args.length * args.mutation_rate)
        
        # get full frequency spectra
        fs0, fs1, jSFS = per_deme_sfs(pop)

        # project to desired sizes
        fs0_proj = fs0.project([2*args.nsam])
        fs1_proj = fs1.project([2*args.nsam])

        # moments spectrum
        theta = 4 * N_ref * args.mutation_rate * args.length
        F = tennessen_moments(args.nsam) * theta
        F0 = F.marginalize([1])
        F1 = F.marginalize([0])
        
        spectra = {'fwdpy':{'Afr':fs0_proj, 'Eur': fs1_proj},
                   'moments':{'Afr':F0, 'Eur': F1}}

        fname = f'/lb/project/gravel/ragsdale_projects/PRS/simulations/spectra_tennessen_ns_{args.nsam}_length_{args.length}_mig_0_seed_{seeds[rep]}.bp'
        with open(fname, 'wb+') as fout:
            pickle.dump(spectra, fout)

