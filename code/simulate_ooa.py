import msprime
import demography
import tskit
import networkx as nx

def ooa(params=None, Ne=7300):
    if params is None:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = [2.11, 0.377, 0.251, 0.111, 0.224, 3.02, 0.0904, 5.77,
                      0.0711, 3.80, 0.256, 0.125, 1.07]
        # gutenkunst params
        # (1.685, 0.219, 0.288, 0.325, 0.137, 4.07, 0.0699, 7.41,
        #    0.0581, 3.65, 0.438, 0.277, 1.40)
    else:
        (nuA, TA, nuB, TB, nuEu0, nuEuF, nuAs0, nuAsF, TF, mAfB, mAfEu, mAfAs,
            mEuAs) = params

    G = nx.DiGraph()
    G.add_node('root', nu=1, T=0)
    G.add_node('A', nu=nuA, T=TA)
    G.add_node('B', nu=nuB, T=TB,
               m={'YRI': mAfB})
    G.add_node('YRI', nu=nuA, T=TB+TF,
               m={'B': mAfB, 'CEU': mAfEu, 'CHB': mAfAs})
    G.add_node('CEU', nu0=nuEu0, nuF=nuEuF, T=TF,
               m={'YRI': mAfEu, 'CHB': mEuAs})
    G.add_node('CHB', nu0=nuAs0, nuF=nuAsF, T=TF,
               m={'YRI': mAfAs, 'CEU': mEuAs})
    edges = [('root', 'A'), ('A', 'B'), ('A', 'YRI'), ('B', 'CEU'),
             ('B', 'CHB')]
    G.add_edges_from(edges)
    dg = demography.DemoGraph(G, Ne=Ne)
    return dg


def get_tree_sequeneces(Ne=7300, pop_ids=['YRI','CEU','CHB'],
                        sample_sizes=[1000, 1000, 1000],
                        recombination_map=None,
                        mutation_rate=2e-8,
                        model='hudson',
                        return_ts=True):
    dg = ooa(Ne=Ne)
    assert recombination_map is not None, "need to give recombination map path"
    
    if return_ts is False:
        print("returning msprime inputs")
        pop_config, mig_mat, demo_events = dg.msprime_inputs(Ne=Ne)
        samples = dg.msprime_samples(pop_ids, sample_sizes)
    else:
        print("simulating tree sequence")
        ts = dg.simulate_msprime(model=model, Ne=Ne, pop_ids=pop_ids,
                                 sample_sizes=sample_sizes,
                                 recombination_map=recombination_map)
        if mutation_rate is not None:
            ts = msprime.mutate(ts, mutation_rate)
        return ts

