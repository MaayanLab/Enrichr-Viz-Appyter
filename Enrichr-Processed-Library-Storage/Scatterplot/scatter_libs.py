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
with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/datasetStatistics') as f:
    stats = json.load(f)
    libs = [l['libraryName'] for l in stats['statistics']]

# ARCHS4 co-expression dataset can be downloaded from the ARCHS4 site
# (https://maayanlab.cloud/archs4/download.html) under the section 
# "Gene Correlation"

# narrow down genes that have co-expression data in ARCHS4
with open('archs4_genes.txt', 'r') as f_in:
    archs4_genes = [g.strip() for g in f_in.readlines()]

def augment_archs4(geneset):
    '''
    Augment a list of unique genes {geneset} with ARCHS4 co-expression data. 
    Sum the Pearson correlation scores of each gene in ARCHS4 co-expression 
    matrix for the genes in {geneset}, excluding the genes already in {geneset},
    and append the top co-expressed genes to {geneset}. Returns new list. 
    '''
    add_len = 500 - len(geneset)
    subset = list(set(geneset).intersection(set(archs4_genes)))
    df = feather.read_feather('human_correlation_archs4.f', columns=subset)
    df = df.set_index(pd.Index(archs4_genes))
    df['sum'] = df.sum(axis=1)
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
    raw_library_data = []
    library_data = []

    with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=' + lib) as f:
        for line in f.readlines():
                raw_library_data.append(line.decode("utf-8").split("\t\t"))

    name = []
    gene_list = []
    aug_gene_list = []

    for i in range(len(raw_library_data)):
        name += [raw_library_data[i][0]]
        raw_genes = [gene.strip() for gene in raw_library_data[i][1].split('\t')]
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
    
    # handle any errors that may arise without pausing processing
    try:
        print("\ttfidf") # keep track of processing step
        tfidf_vectorizer = TfidfVectorizer(
            analyzer=lambda gene: gene,
            min_df = 3,
            max_df = 0.05,
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
    except:
        print("something went wrong with", lib, '-- continuing')
        continue

