import urllib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import umap.umap_ as umap
from os import path
from maayanlab_bioinformatics.enrichment import enrich_crisp
import json
import pyarrow.feather as feather

# get list of all libraries in Enrichr 
# with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/datasetStatistics') as f:
#     stats = json.load(f)
#     libs = [l['libraryName']) for l in stats['statistics']]

# use list of specific library names; best if processing local libraries
libs = [
    # INSERT LIBRARIES HERE
]

# ARCHS4 co-expression dataset can be downloaded from the ARCHS4 site
# (https://maayanlab.cloud/archs4/download.html) under the section 
# "Gene Correlation"

# extract list of genes that have co-expression data in ARCHS4
with open('archs4_data/archs4_genes.txt', 'r') as f_in:
    archs4_genes = [g.strip() for g in f_in.readlines()]

archs4_df = feather.read_feather('archs4_data/human_correlation_archs4.f')
archs4_df.index = archs4_df.columns

def augment_archs4(geneset):
    '''
    Augment a list of unique genes {geneset} with ARCHS4 co-expression data. 
    Sum the Pearson correlation scores of each gene in ARCHS4 co-expression 
    matrix for the genes in {geneset}, excluding the genes already in {geneset},
    and append the top co-expressed genes to {geneset}. Returns new list. 
    '''
    # only augment to ~500 genes for efficiency's sake
    if len(geneset) >= 500:
        return geneset
    add_len = 500 - len(geneset)

    # only look for genes in geneset with ARCHS4 co-expression data
    subset = list(set(geneset).intersection(set(archs4_genes)))
    
    # read only data columns for genes in geneset
    df = archs4_df.loc[archs4_genes, subset]

    # sum co-expression values for all genes, for each gene in geneset
    df['sum'] = df.sum(axis=1)

    # get genes with highest summed co-exp and append to original geneset
    df = df[df['sum'] > 0].sort_values(by='sum', ascending=False)
    return geneset + df.index.tolist()[:min(add_len, df.shape[0])]


def get_Enrichr_library(lib):
    '''
    Processes the GMT file for the input Enrichr library {lib} and returns a 
    list of lists: 
    [
        ["gene set name", [original gene set], [augmented gene set]], 
        [], 
        ...
    ]
    '''
    # variables to store data
    raw_library_data = []
    library_data = []

    # # get library data (GMT file) from Enrichr
    # with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=' + lib) as f:
    #     for line in f.readlines():
    #           raw_library_data.append(line.decode("utf-8").split("\t\t"))

    # get library data (GMT file) locally
    with open(f'../../../Libs_to_scatter/{lib}.gmt', 'r') as f:
        for line in f.readlines():
            raw_library_data.append(line.split("\t\t"))

    # keep track of geneset data
    name = []
    gene_list = []
    aug_gene_list = []
    
    for i in range(len(raw_library_data)):
        name += [raw_library_data[i][0]]
        raw_genes = [gene.strip().split(',')[0] for gene in raw_library_data[i][1].split('\t')]
        gene_list += [raw_genes[:-1]]
        
        # augment with ARCHS4 coexpression data
        aug_genes = augment_archs4(raw_genes[:-1])
        aug_gene_list += [aug_genes]

    library_data = [list(a) for a in zip(name, gene_list, aug_gene_list)]
    return library_data


for lib in libs:
    if path.exists('Libraries/' + lib + '.csv'):
        continue
    print("SCATTER LIB:", lib) # keep track of library

    # handle library inaccessibility
    try:
        library_data = get_Enrichr_library(lib)
    except:
        print("failed to access", lib, "-- continuing")
        continue

    df = pd.DataFrame(
        data = library_data, 
        columns = ['Name', 'Genes', 'Augmented_Genes']
    )

    gene_list = df['Augmented_Genes']
    
    print("\ttfidf") # keep track of processing step
    try:
        tfidf_vectorizer = TfidfVectorizer(
            analyzer=lambda gene: gene,
            min_df = 3,
            max_df = 0.05,
            max_features = 100000,
            ngram_range=(1, 1)
        )
        tfidf = tfidf_vectorizer.fit_transform(gene_list)

    except:
        tfidf_vectorizer = TfidfVectorizer(
            analyzer=lambda gene: gene,
            min_df = 3,
            max_df = 0.25,
            max_features = 100000,
            ngram_range=(1, 1)
        )
        tfidf = tfidf_vectorizer.fit_transform(gene_list)

    print("\tumap") # keep track of processing step
    reducer = umap.UMAP()
    reducer.fit(tfidf)
    embedding = pd.DataFrame(reducer.transform(tfidf), columns=['x','y'])

    df['Genes'] = df['Genes'].apply(lambda x: ' '.join(x))
    pd.concat([embedding, df[['Name', 'Genes']]], axis=1).to_csv('Libraries/' + lib + '.csv', index = False)

