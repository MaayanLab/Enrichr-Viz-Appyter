import pandas as pd 
import numpy as np
import pickle
import json
import requests
import math
import matplotlib
import uuid
import urllib
import time as time
from os import path, listdir
from textwrap import dedent
from IPython.core.display import display, HTML
from string import Template
from random import seed, randint
from operator import itemgetter

all_libraries = [
    # INSERT LIBRARIES TO BE ADDED HERE
]

def library_processing(library_index):
    # processes library data
    raw_library_data = []
    # library_data = []

    with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=' + all_libraries[library_index]) as f:
        for line in f.readlines():
                raw_library_data.append(line.decode("utf-8").split("\t\t"))
    name = []
    gene_list = []

    for i in range(len(raw_library_data)):
        name += [raw_library_data[i][0]]
        raw_genes = raw_library_data[i][1].split("\t")
        gene_list += [raw_genes[:-1]]

    # determine the dimensions of the canvas
    x_dimension = math.ceil(math.sqrt(len(name)))
    y_dimension = math.ceil(math.sqrt(len(name)))

    # zip name, gene_list, indices, and blank list for neighbor score then add dummy entries to the zipped list
    num_hex = x_dimension*y_dimension
    neighbor_score = [0.0] * len(name)
    anneal_list = list(zip(name, gene_list, neighbor_score))

    # add "dummy" hexagons so the rectangular shape is filled
    for i in range(len(name), num_hex):
        anneal_list += [('', [], 0.0)]

    return anneal_list, x_dimension, y_dimension


def unzip_list(anneal_list):
    unzipped_list = zip(*anneal_list)
    return list(unzipped_list)


def find_neighbors(ind, x_dimension, y_dimension):
    # returns a list of the indices of the neighbors of the index given
    
    neighbors = []
    num_hex = x_dimension * y_dimension

    if 0 <= ind <= x_dimension-1:
        # top row (inc. corners)
        if ind == 0:
            # top left corner
            neighbors = [num_hex-1, num_hex-x_dimension, x_dimension-1, 2*x_dimension-1, ind+1, ind+x_dimension]
        elif ind == (x_dimension-1):
            # top right corner
            neighbors = [ind-1, ind+x_dimension, 0, ind+x_dimension-1, num_hex-2, num_hex-1]
        else:
            # non-corner top row
            neighbors = [ind-1, ind+1, ind+x_dimension, ind+num_hex-x_dimension-1, 
            ind+num_hex-x_dimension, ind+x_dimension-1]

    elif (num_hex - x_dimension) <= ind <= num_hex -1:
        if ind == (num_hex-x_dimension):
            # bottom left corner
            neighbors = [ind+1, ind-x_dimension, ind-x_dimension+1, 0, 1, num_hex-1]
        elif ind == (num_hex-1):
            # bottom right corner
            neighbors = [ind-1, ind-x_dimension, ind-x_dimension+1, 0, x_dimension-1,
            num_hex-2*x_dimension]
        else:
            # non-corner bottom row
            neighbors = [ind-1, ind+1, ind-x_dimension, ind-x_dimension+1, ind-num_hex+x_dimension,
            ind-num_hex+x_dimension+1]
    elif ind % y_dimension == 0 and (ind/y_dimension)%2 == 1:
        # "inner" left edge (not top or bottom row)
        neighbors = [ind+x_dimension-1, ind+1, ind-x_dimension, ind-x_dimension+1, ind+x_dimension, 
        ind+x_dimension+1]
    elif ind % y_dimension == 0 and (ind/y_dimension)%2 == 0:
        # "outer" left edge (not top or bottom row)
        neighbors = [ind-1, ind+1, ind+x_dimension, ind+2*x_dimension-1, ind-x_dimension, 
        ind+x_dimension-1]
    elif (ind+1) % y_dimension == 0 and ((ind+1)/y_dimension)%2 == 0:
        # "outer" right edge (not top or bottom row)
        neighbors = [ind-1, ind+1, ind-x_dimension, ind-x_dimension+1, ind+x_dimension, 
        ind-2*x_dimension+1]
    elif (ind+1) % y_dimension == 0 and ((ind+1)/y_dimension)%2 == 1:
        # "inner" right edge (not top or bottom row)
        neighbors = [ind-1, ind-x_dimension-1, ind-x_dimension, ind-x_dimension+1, ind+x_dimension, 
        ind+x_dimension-1]
    else:
        # middle
        neighbors = [ind-1, ind+1, ind-x_dimension, ind-x_dimension+1, ind+x_dimension, 
        ind+x_dimension+1]
    return neighbors


# initially find fitness
def find_fitness(anneal_list, x_dimension, y_dimension):
    fitness = 0
    for i in range(len(anneal_list)):
        neighbors = find_neighbors(i, x_dimension, y_dimension)
        sum_neighbor_score = 0
        for index in neighbors:
            intersection = [value for value in anneal_list[index][1] if value in anneal_list[i][1]]
            if len(anneal_list[index][1]) + len(anneal_list[i][1]) != 0:
                jaccard = len(intersection)/(len(anneal_list[index][1]) + len(anneal_list[i][1]))
            else:
                jaccard = 0.0
            sum_neighbor_score += jaccard
        hex_list = list(anneal_list[i])
        hex_list[2] = sum_neighbor_score
        hex_tuple = tuple(hex_list)
        anneal_list[i] = hex_tuple
        fitness += sum_neighbor_score
    return fitness, anneal_list


# take indices of swapped hexagons
def find_swapped_fitness(anneal_list, swapped_a, swapped_b, old_fitness, x_dimension, y_dimension):
    neighbors_a = find_neighbors(swapped_a, x_dimension, y_dimension)
    neighbors_b = find_neighbors(swapped_b, x_dimension, y_dimension)
    hexagons_to_update = [swapped_a, swapped_b] + neighbors_a + neighbors_b
    anneal_copy = anneal_list.copy()

    new_fitness = 0
    # Recalculate scores for all hexagons that need updating
    for hex in hexagons_to_update:

        # subtract out the swapped neighbor fitnesses because they are changing 
        old_fitness -= anneal_copy[hex][2]

        neighbors = find_neighbors(hex, x_dimension, y_dimension)
        sum_neighbor_score = 0
        for index in neighbors:
            intersection = [value for value in anneal_copy[index][1] if value in anneal_copy[hex][1]]
            if len(anneal_copy[index][1]) + len(anneal_copy[hex][1]) != 0:
                jaccard = len(intersection)/(len(anneal_copy[index][1]) + len(anneal_copy[hex][1]))
            else:
                jaccard = 0.0
            sum_neighbor_score += jaccard
        hex_list = list(anneal_copy[hex])
        hex_list[2] = sum_neighbor_score
        hex_tuple = tuple(hex_list)
        anneal_copy[hex] = hex_tuple
        new_fitness += sum_neighbor_score
        
    return old_fitness + new_fitness, anneal_copy


def annealing(anneal_list, steps, old_fitness, x_dimension, y_dimension):
    num_hex = x_dimension * y_dimension
    # returns unzipped list
    for i in range(steps):
        index_a = randint(0, num_hex-1)
        index_b = randint(0, num_hex-1)
        anneal_list[index_a], anneal_list[index_b] = anneal_list[index_b], anneal_list[index_a]
        new_fitness, new_anneal_list = find_swapped_fitness(anneal_list, index_a, index_b, old_fitness, x_dimension, y_dimension)

        if new_fitness <= old_fitness:
            # swap back
            anneal_list[index_a], anneal_list[index_b] = anneal_list[index_b], anneal_list[index_a]
        else:
            # finalize the swap by resetting old_fitness and changing anneal_list
            old_fitness = new_fitness
            anneal_list = new_anneal_list
    return anneal_list


def fileConversion(library_name):
    print("now converting: " + library_name)
    library_data = []

    with open('Annealed-Libraries/' + library_name + '.txt', 'rb') as f:
        library_data = pickle.load(f)

    with open('Annealed-Libraries/' + library_name + '.txt', 'w') as f:
        for index in range(len(library_data)):
            new_line = ''
            new_line += library_data[index][0] + '\t' + '\t'
            for gene in library_data[index][1]:
                new_line += gene + '\t'
            new_line += '\n'
            f.write(new_line)


# save library files
print('\nProcessing Libraries...\n')
for library_index in range(len(all_libraries)):
    if path.exists('Annealed-Libraries/' + all_libraries[library_index] + '.txt'):
        continue
    t = time.time()
    anneal_list, x_dimension, y_dimension = library_processing(library_index)
    anneal_list = annealing(anneal_list, 100000, find_fitness(anneal_list, x_dimension, y_dimension)[0], x_dimension, y_dimension)
    unzipped_anneal_list = unzip_list(anneal_list)
    processed_list = list(zip(unzipped_anneal_list[0], unzipped_anneal_list[1]))
    with open('Annealed-Libraries/' + all_libraries[library_index] + '.txt', 'wb+') as f:
        pickle.dump(processed_list, f)
    print(all_libraries[library_index], 'processed in', time.time()-t, 'sec')

# convert library files
print('\nConverting Files...\n')
for library_name in all_libraries:
    fileConversion(library_name)
    print(library_name, 'successfully converted!')


''' 
existing libraries:
    'Descartes_Cell_Types_and_Tissue_2021',
    'COVID-19_Related_Gene_Sets',
    'Enrichr_Users_Contributed_Lists_2020',
    'MSigDB_Hallmark_2020',
    'TG_GATES_2020'
    'ARCHS4_TFs_Coexp',
    'ChEA_2016',
    'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X',
    'ENCODE_Histone_Modifications_2015',
    'ENCODE_TF_ChIP-seq_2015',
    'Epigenomics_Roadmap_HM_ChIP-seq',
    'Enrichr_Submissions_TF-Gene_Coocurrence',
    'Genome_Browser_PWMs',
    'lncHUB_lncRNA_Co-Expression',
    'miRTarBase_2017',
    'TargetScan_microRNA_2017',
    'TF-LOF_Expression_from_GEO',
    'TF_Perturbations_Followed_by_Expression',
    'TRANSFAC_and_JASPAR_PWMs',
    'ARCHS4_Kinases_Coexp',
    'BioPlanet_2019',
    'BioPlex_2017',
    'CORUM',
    'Elsevier_Pathway_Collection',
    'Kinase_Perturbations_from_GEO_down',
    'Kinase_Perturbations_from_GEO_up',
    'L1000_Kinase_and_GPCR_Perturbations_down',
    'L1000_Kinase_and_GPCR_Perturbations_up',
    'NURSA_Human_Endogenous_Complexome',
    'PPI_Hub_Proteins',
    'Reactome_2016',
    'Virus-Host_PPI_P-HIPSTer_2020',
    'WikiPathways_2019_Mouse',
    'Human_Phenotype_Ontology',
    'Jensen_COMPARTMENTS',
    'Jensen_DISEASES',
    'Jensen_TISSUES',
    'MGI_Mammalian_Phenotype_Level_4_2019',  
    'ARCHS4_IDG_Coexp',
    'DepMap_WG_CRISPR_Screens_Broad_CellLines_2019',
    'DepMap_WG_CRISPR_Screens_Sanger_CellLines_2019',
    'DisGeNET',
    'DrugMatrix',
    'DSigDB',
    'GeneSigDB',
    'GWAS_Catalog_2019',
    'LINCS_L1000_Chem_Pert_down',
    'LINCS_L1000_Chem_Pert_up',
    'Old_CMAP_down',
    'Old_CMAP_up',
    'Rare_Diseases_AutoRIF_ARCHS4_Predictions',
    'Rare_Diseases_AutoRIF_Gene_Lists',
    'Rare_Diseases_GeneRIF_ARCHS4_Predictions',
    'Rare_Diseases_GeneRIF_Gene_Lists',
    'Virus_Perturbations_from_GEO_down',
    'Virus_Perturbations_from_GEO_up',
    'Allen_Brain_Atlas_down',
    'Allen_Brain_Atlas_up',
    'ARCHS4_Cell-lines',
    'ARCHS4_Tissues',
    'Cancer_Cell_Line_Encyclopedia',
    'CCLE_Proteomics_2020',
    'ESCAPE',
    'GTEx_Tissue_Sample_Gene_Expression_Profiles_down',
    'GTEx_Tissue_Sample_Gene_Expression_Profiles_up',
    'Human_Gene_Atlas',
    'Mouse_Gene_Atlas',
    'NCI-60_Cancer_Cell_Lines',
    'ProteomicsDB_2020',        
    'Chromosome_Location_hg19',
    'Genes_Associated_with_NIH_Grants',
    'HMDB_Metabolites',
    'HomoloGene',
    'InterPro_Domains_2019',
    'NIH_Funded_PIs_2017_AutoRIF_ARCHS4_Predictions',
    'NIH_Funded_PIs_2017_GeneRIF_ARCHS4_Predictions',
    'NIH_Funded_PIs_2017_Human_AutoRIF',
    'NIH_Funded_PIs_2017_Human_GeneRIF',
    'Table_Mining_of_CRISPR_Studies',
    'ChEA_2013',
    'ChEA_2015',
    'Chromosome_Location',
    'Disease_Signatures_from_GEO_down_2014',
    'Disease_Signatures_from_GEO_up_2014',
    'Drug_Perturbations_from_GEO_2014',
    'ENCODE_Histone_Modifications_2013',
    'ENCODE_TF_ChIP-seq_2014',
    'GO_Biological_Process_2013',
    'GO_Biological_Process_2015',
    'GO_Biological_Process_2017',
    'GO_Biological_Process_2017b',
    'GO_Cellular_Component_2013',
    'GO_Cellular_Component_2015',
    'GO_Cellular_Component_2017',
    'GO_Cellular_Component_2017b',
    'GO_Molecular_Function_2013',
    'GO_Molecular_Function_2015',
    'GO_Molecular_Function_2017',
    'GO_Molecular_Function_2017b',
    'HumanCyc_2015',
    'KEA_2013',
    'KEGG_2013',
    'KEGG_2015',
    'KEGG_2016',
    'MGI_Mammalian_Phenotype_2013',
    'MGI_Mammalian_Phenotype_2017',
    'MGI_Mammalian_Phenotype_Level_3',
    'MGI_Mammalian_Phenotype_Level_4',
    'NCI-Nature_2015',
    'Panther_2015',
    'Reactome_2013',
    'Reactome_2015',
    'TargetScan_microRNA',
    'Tissue_Protein_Expression_from_ProteomicsDB',
    'WikiPathways_2013',
    'WikiPathways_2015',
    'WikiPathways_2016',               
    'Aging_Perturbations_from_GEO_down',
    'Aging_Perturbations_from_GEO_up',
    'Disease_Perturbations_from_GEO_down',
    'Disease_Perturbations_from_GEO_up',
    'Drug_Perturbations_from_GEO_down',
    'Drug_Perturbations_from_GEO_up',
    'Gene_Perturbations_from_GEO_down',
    'Gene_Perturbations_from_GEO_up',
    'Ligand_Perturbations_from_GEO_down',
    'Ligand_Perturbations_from_GEO_up',
    'MCF7_Perturbations_from_GEO_down',
    'MCF7_Perturbations_from_GEO_up',
    'Microbe_Perturbations_from_GEO_down',
    'Microbe_Perturbations_from_GEO_up',
    'RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO',
    'SysMyo_Muscle_Gene_Sets'
'''