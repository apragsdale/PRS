"""
takes tree sequence with mutations, assign effect sizes,
run gwas, with clumping, and then infer prs, comparing to
true prs

taken from alicia's code
"""

import numpy as np
import msprime
import tskit
import gzip
from datetime import datetime
import random
import os, sys, math
from tqdm import tqdm
from scipy import stats


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def current_time():
    return(' [' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']')


def true_prs(ts, ncausal, h2, nhaps, out):
    """
    ncausal : int, number of true causal alleles
    h2 : float, snp heritability
    nhaps : list, [ns_afr, ns_eur, ns_eas]
    out : 
    
    For the ncausal alleles, assign them effect sizes based off of h^2
    From these effect sizes, compute polygenic risk scores for everyone
    """
    eprint('Reading all site info' + current_time())
    
    ### alicia had them evenly spaced across genome. randomly assign instead
    #causal_mut_index = np.linspace(0, ts.get_num_mutations()-1, 
    #                               ncausal, dtype=int)
    causal_mut_index = np.random.choice(range(ts.get_num_mutations()),
                                        size=ncausal, replace=False)
    causal_mutations = set()
    
    # go through each population's trees
    out_sites = gzip.open(out + '_nhaps_' + '_'.join(map(str, nhaps)) + '_h2_'
                          + str(round(h2, 2)) + '_m_' + str(ncausal) + 
                          '.sites.gz', 'wb+')
    out_sites.write(('\t'.join(['Index', 'Pos', 'AFR_count', 'EUR_count', 
                                'EAS_count', 'Total', 'beta'])
                     + '\n').encode())
    mut_info = {} # index -> position, afr count, eur count, eas count
    pop_count = 0
    
    # loop through each population, get position and allele frequencies for
    # each causal mutation
    for pop_leaves in [ts.get_samples()[:nhaps[0]],
                       ts.get_samples()[nhaps[0]:nhaps[0]+nhaps[1]],
                       ts.get_samples()[nhaps[0]+nhaps[1]:]]:
#                      [ts.get_samples(population_id=0),
#                       ts.get_samples(population_id=1),
#                       ts.get_samples(population_id=2)]:
        for tree in ts.trees(tracked_leaves=pop_leaves):
            for mutation in tree.mutations():
                if mutation.index in causal_mut_index:
                    causal_mutations.add(mutation.index)
                    if pop_count == 0:
                        mut_info[mutation.index] = [mutation.position,
                            tree.get_num_tracked_leaves(mutation.node)]
                    else:
                        mut_info[mutation.index].append(
                            tree.get_num_tracked_leaves(mutation.node))
        pop_count += 1
    
    # assign causal effects as a random normal with mean zero and variance
    # h^2/M, where M is the number of causal mutations
    causal_effects = {}
    for mutation_index in causal_mutations:
        causal_effects[mutation_index] = np.random.normal(loc=0,scale=h2/ncausal)
    
    
    eprint('Writing all site info' + current_time())
    for mutation in sorted(causal_mutations):
        out_sites.write((str(mutation) + '\t' + 
                         '\t'.join(map(str, mut_info[mutation])) +
                         '\t' + str(ts.get_sample_size()) + '\t').encode())
        out_sites.write((str(causal_effects[mutation]) + '\n').encode())
    out_sites.close()
    
    prs_haps = np.zeros(sum(nhaps)) #score for each haplotype
    eprint('Computing true PRS' + current_time())
    for variant in tqdm(ts.variants(), total=ts.get_num_mutations()):
        if variant.index in causal_mut_index:
            prs_haps += variant.genotypes * causal_effects[variant.index] # multiply vector of genotypes by beta for given variant
    prs_true = prs_haps[0::2] + prs_haps[1::2] #add to get individuals
    return(prs_true)
    
def case_control(prs_true, h2, nhaps, prevalence, ncontrols):
    """
    get cases assuming liability threshold model
    get controls from non-cases in same ancestry
    """
    eprint('Defining cases/controls' + current_time())
    env_effect = np.random.normal(loc=0,scale=1-h2, size=sum(nhaps)//2)
    prs_norm = (prs_true - np.mean(prs_true)) / np.std(prs_true)
    env_norm = (env_effect - np.mean(env_effect)) / np.std(env_effect)
    total_liability = math.sqrt(h2) * prs_norm + math.sqrt(1 - h2) * env_norm
    eur_liability = total_liability[nhaps[0]//2:(nhaps[0]+nhaps[1])//2]
    sorted_liability = sorted(eur_liability)
    cases = [i for (i, x) in enumerate(eur_liability) if x >= sorted_liability[int((1-prevalence)*len(sorted_liability))]]
    controls = set(range(nhaps[1]//2))
    for case in cases:
        controls.remove(case)
    controls = random.sample(controls, ncontrols)
    
    case_ids = [x+nhaps[0]//2 for x in cases]
    control_ids = [x+nhaps[0]//2 for x in sorted(controls)]
    
    return case_ids, control_ids, prs_norm, env_norm

def run_gwas(ts, diploid_cases, diploid_controls, p_threshold, cc_maf):
    """
    use cases and controls to compute OR, log(OR), and p-value for every variant
    """
    eprint('Running GWAS (' + str(len(diploid_cases)) + ' cases, '
           + str(len(diploid_controls)) + ' controls)' + current_time())
    summary_stats = {} # position -> OR, p-value
    case_control = {} # position -> ncases w mut, ncontrols w mut
    
    cases = [2*x for x in diploid_cases] + [2*x+1 for x in diploid_cases]
    controls = [2*x for x in diploid_controls] + [2*x+1 for x in diploid_controls]
    
    eprint('Counting case mutations' + current_time())
    for tree in ts.trees(tracked_leaves=cases):
        for mutation in tree.mutations():
            case_control[mutation.position] = [tree.get_num_tracked_leaves(mutation.node)]
            
    eprint('Counting control mutations' + current_time())
    for tree in ts.trees(tracked_leaves=controls):
        for mutation in tree.mutations():
            case_control[mutation.position].append(
                tree.get_num_tracked_leaves(mutation.node))
    
    # only keep sites with non-infinite or nan effect size with case and control maf > .01
    num_var = 0
    eprint('Computing fisher\'s exact test' + current_time())
    num_controls = float(len(controls))
    num_cases = float(len(cases))
    for position in tqdm(case_control):
        case_maf = min(case_control[position][0]/num_cases, 
                       (num_cases - case_control[position][0])/num_cases)
        control_maf = min(case_control[position][1]/num_controls,
                          (num_controls - case_control[position][1])/num_controls)
        case_control_maf = min((case_control[position][0]+case_control[position][1])/(num_cases+num_controls), (num_cases + num_controls - case_control[position][0] - case_control[position][1])/(num_cases + num_controls))
        if case_control_maf > cc_maf:
            contingency = [[case_control[position][0], num_cases - case_control[position][0]],
                [case_control[position][1], num_controls - case_control[position][1]]]
            (OR, p) = stats.fisher_exact(contingency) #OR, p-value
            if not np.isnan(OR) and not np.isinf(OR) and OR != 0 and p <= p_threshold:
                summary_stats[position] = [OR, p]
                num_var += 1
            
    eprint('Done with GWAS! (' + str(len(summary_stats)) + ' amenable sites)'
           + current_time())  
    return(summary_stats, cases, controls)

def clump_variants(ts, summary_stats, nhaps, r2_threshold, window_size):
    """
    perform variant clumping in a greedy fasion with p-value and r2 threshold in windows
    return only those variants meeting some nominal threshold
    
    1: make a dict of pos -> variant for subset of sites meeting criteria
    2: make an r2 dict of all pairs of snps meeting p-value threshold and in same window
    """
    # make a list of SNPs ordered by p-value
    eprint('Subsetting variants to usable list' + current_time())
    usable_positions = {} # position -> variant (ts indices)
    
    sim_pos_index = {}
    for variant in tqdm(ts.variants(), total=ts.get_num_mutations()):
        if variant.position in summary_stats:
            usable_positions[variant.position] = variant
            sim_pos_index[variant.position] = variant.index
    
    # order all snps by p-value
    ordered_positions = sorted(summary_stats.keys(), key=lambda x: summary_stats[x][-1])
    #[(x, (x in usable_positions.keys())) for x in ordered_positions]
    
    eur_subset = ts.simplify(range(nhaps[0], (nhaps[0]+nhaps[1])))
    eur_index_pos = {}
    eur_pos_index = {}
    for mutation in tqdm(eur_subset.mutations(), total=eur_subset.get_num_mutations()):
        eur_index_pos[mutation.index] = mutation.position
        eur_pos_index[mutation.position] = mutation.index
    ordered_eur_index = sorted(eur_index_pos.keys())
    ld_calc = msprime.LdCalculator(eur_subset)
    #ld_calc = msprime.LdCalculator(ts)
    
    # compute LD and prune in order of significance (popping index of SNPs)
    for position in ordered_positions:
        if position in usable_positions:
            r2_forward = ld_calc.get_r2_array(eur_pos_index[position], direction=msprime.FORWARD, max_distance=125e3)
            for i in np.where(r2_forward > r2_threshold)[0]:
                usable_positions.pop(eur_index_pos[eur_pos_index[position]+i+1],
                                     None) #identify next position in eur space
            r2_reverse = ld_calc.get_r2_array(eur_pos_index[position], direction=msprime.REVERSE, max_distance=125e3)
            for i in np.where(r2_reverse > r2_threshold)[0]:
                usable_positions.pop(eur_index_pos[eur_pos_index[position]-i-1],
                                     None)
    
    clumped_snps = set(usable_positions.keys())
    
    eprint('Starting SNPs: ' + str(len(ordered_positions))
           + '; SNPs after clumping: '
           + str(len(clumped_snps)) + current_time())

    return(clumped_snps, usable_positions)
    

def infer_prs(ts, nhaps, clumped_snps, summary_stats, usable_positions, h2,
              ncausal, out):
    """
    use clumped variants from biased gwas to compute inferred prs for everyone
    """
    eprint('Computing inferred PRS' + current_time())
    prs_haps = np.zeros(sum(nhaps))
    for variant in tqdm(ts.variants(), total=ts.get_num_mutations()):
        if variant.position in usable_positions:
            for ind in range(ts.get_sample_size()):
                prs_haps[ind] += int(variant.genotypes[ind]) * math.log(summary_stats[variant.position][0])
    
    prs_infer = prs_haps[0::2] + prs_haps[1::2]
    
    # go through each population's trees
    out_sites = gzip.open(out + '_nhaps_' + '_'.join(map(str, nhaps)) + '_h2_'
                          + str(round(h2, 2)) + '_m_' + str(ncausal) +
                          '.infer_sites.gz', 'wb+')
    out_sites.write(('\t'.join(['Index', 'Pos', 'AFR_count', 'EUR_count',
                                'EAS_count', 'Total', 'beta']) + '\n').encode())
    mut_info = {}
    causal_mutations = set()
    pop_count = 0
    for pop_leaves in [ts.get_samples()[:nhaps[0]],
                       ts.get_samples()[nhaps[0]:nhaps[0]+nhaps[1]],
                       ts.get_samples()[nhaps[0]+nhaps[1]:]]:
#                      [ts.get_samples(population_id=0),
#                       ts.get_samples(population_id=1),
#                       ts.get_samples(population_id=2)]:
        for tree in ts.trees(tracked_leaves=pop_leaves):
            for mutation in tree.mutations():
                if mutation.position in usable_positions:
                    causal_mutations.add(mutation.index)
                    if pop_count == 0:
                        mut_info[mutation.index] = [mutation.position, tree.get_num_tracked_leaves(mutation.node)]
                    else:
                        mut_info[mutation.index].append(tree.get_num_tracked_leaves(mutation.node))
        pop_count += 1
    
    eprint('Writing all site info' + current_time())
    for mutation in sorted(causal_mutations):
        out_sites.write((str(mutation) + '\t'
                        + '\t'.join(map(str, mut_info[mutation]))
                        + '\t' + str(ts.get_sample_size()) + '\t').encode())
        out_sites.write((str(summary_stats[ts.tables.sites[mutation].position][0]) + '\n').encode())
        #
    out_sites.close()
    
    return(prs_infer)
    
def write_summaries(out, prs_true, prs_infer, nhaps, cases, controls, h2, ncausal, environment):
    eprint('Writing output!' + current_time())
    scaled_prs = math.sqrt(h2) * prs_true
    scaled_env = math.sqrt(1 - h2) * environment
    out_prs = gzip.open(out + '_nhaps_' + '_'.join(map(str, nhaps)) 
                        + '_h2_' + str(round(h2, 2)) + '_m_' + str(ncausal)  
                        + '.prs.gz', 'wb')
    out_prs.write(('\t'.join(['Ind', 'Pop', 'PRS_true', 'PRS_infer', 'Pheno',
                              'Environment']) + '\n').encode())
    for ind in range(len(prs_true)):
        if ind in cases:
            pheno = 1
        elif ind in controls:
            pheno = 0
        else:
            pheno = 'NA'
        if ind in range(nhaps[0]/2):
            pop = 'AFR'
        elif ind in range(nhaps[0]/2, nhaps[0]/2+nhaps[1]/2):
            pop = 'EUR'
        else:
            pop = 'EAS'
        out_prs.write(('\t'.join(map(str, [ind+1, pop, prs_true[ind],
                                           prs_infer[ind], pheno,
                                           scaled_env[ind]]))
                                           + '\n').encode())
    out_prs.close()


