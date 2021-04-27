import urllib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import umap.umap_ as umap
import itertools
from os import path

from maayanlab_bioinformatics.enrichment import enrich_crisp

from sklearn.decomposition import NMF

libs = [
    # INSERT LIBRARIES TO BE ADDED HERE
]

def get_Enrichr_library(lib):
    # processes library data
    raw_library_data = []
    library_data = []

    with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=' + lib) as f:
        for line in f.readlines():
                raw_library_data.append(line.decode("utf-8").split("\t\t"))

    name = []
    gene_list = []

    for i in range(len(raw_library_data)):
        name += [raw_library_data[i][0]]
        raw_genes = raw_library_data[i][1].replace('\t', ' ')
        gene_list += [raw_genes[:-1]]

    library_data = [list(a) for a in zip(name, gene_list)]
    
    return library_data


for lib in libs:
    if path.exists('Libraries/' + lib + '.csv'):
        continue
    print("SCATTER LIB:", lib)
    try:
        library_data = get_Enrichr_library(lib)
    except:
        print("failed to access", lib, "-- continuing")
        continue

    df = pd.DataFrame(data = library_data, columns = ['Name', 'Genes'])

    gene_list = df['Genes']

    print("\ttfidf")
    tfidf_vectorizer = TfidfVectorizer(
        min_df = 3,
        max_df = 0.005,
        max_features = 100000,
        ngram_range=(1, 1)
    )
    tfidf = tfidf_vectorizer.fit_transform(gene_list)

    print("\tumap")
    reducer = umap.UMAP()
    reducer.fit(tfidf)
    embedding = pd.DataFrame(reducer.transform(tfidf), columns=['x','y'])

    pd.concat([embedding, df], axis=1).to_csv('Libraries/' + lib + '.csv', index = False)


'''
existing libraries:
    'Descartes_Cell_Types_and_Tissue_2021',
    'Chromosome_Location',
    'COVID-19_Related_Gene_Sets',
    'Enrichr_Users_Contributed_Lists_2020',
    'MSigDB_Hallmark_2020',
    'TG_GATES_2020'
    'BioPlex_2017',
    'Virus-Host_PPI_P-HIPSTer_2020',
    'GO_Biological_Process_2018',
    'MGI_Mammalian_Phenotype_Level_4_2019',
    'DisGeNET',
    'DrugMatrix', 
    'DSigDB',
    'LINCS_L1000_Chem_Pert_down',
    'LINCS_L1000_Chem_Pert_up',
    'Old_CMAP_down',
    'Old_CMAP_up',
    'Genes_Associated_with_NIH_Grants',
    'NIH_Funded_PIs_2017_AutoRIF_ARCHS4_Predictions',
    'NIH_Funded_PIs_2017_GeneRIF_ARCHS4_Predictions',
    'NIH_Funded_PIs_2017_Human_AutoRIF',
    'NIH_Funded_PIs_2017_Human_GeneRIF',            
    'BioCarta_2013', 
    'BioCarta_2015', 
    'ChEA_2013',
    'ChEA_2015',
    'Disease_Signatures_from_GEO_down_2014',
    'Disease_Signatures_from_GEO_up_2014',
    'Drug_Perturbations_from_GEO_2014',
    'ENCODE_Histone_Modifications_2013',
    'ENCODE_TF_ChIP-seq_2014',
    'GO_Biological_Process_2013',
    'GO_Biological_Process_2015',
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
    'MGI_Mammalian_Phenotype_2013',
    'MGI_Mammalian_Phenotype_2017',
    'MGI_Mammalian_Phenotype_Level_3',
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