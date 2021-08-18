import urllib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import re
import hashlib
from os import path
import json
import pyarrow.feather as feather

### get list of all libraries in Enrichr 
with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/datasetStatistics') as f:
    stats = json.load(f)
    libs = [(l['libraryName'], l['genesPerTerm']) for l in stats['statistics']]

### use list of specific library names; best if processing local libraries
# libs = [
#     # INSERT LIBRARIES HERE
# ]

### ARCHS4 co-expression dataset can be downloaded from the ARCHS4 site
### (https://maayanlab.cloud/archs4/download.html) under the section 
### "Gene Correlation"

archs4_df = feather.read_feather('archs4_data/human_correlation_archs4.f')
archs4_df.index = archs4_df.columns
archs4_genes = archs4_df.index.tolist()

def augment_archs4(geneset):
    '''
    Augment a list of unique genes {geneset} with ARCHS4 co-expression data. 
    Sum the Pearson correlation scores of each gene in ARCHS4 co-expression 
    matrix for the genes in {geneset}, excluding the genes already in {geneset},
    and append the top co-expressed genes to {geneset}. Returns a list of the
    original genes plus any genes added during augmentation. 
    '''
    ### only augment to ~500 genes for efficiency's sake
    if len(geneset) >= 500:
        return geneset
    add_len = 500 - len(geneset)

    ### only look for genes in geneset with ARCHS4 co-expression data
    subset = list(set(geneset).intersection(set(archs4_genes)))
    
    ### read only data columns for genes in geneset
    df = archs4_df.loc[archs4_genes, subset]

    ### sum co-expression values for all genes, for each gene in geneset
    df['sum'] = df.sum(axis=1)

    ### get genes with highest summed co-exp and append to original geneset
    df = df[df['sum'] > 0].sort_values(by='sum', ascending=False)
    return geneset + df.index.tolist()[:min(add_len, df.shape[0])]


def get_Enrichr_library(lib, local=False, augmented=False):
    '''
    Processes the GMT file for the input Enrichr library {lib} and returns a 
    dictionary where the keys correspond to gene set names, and the value for
    each key is a space-delimited string containing all genes belonging to
    the gene set: 
    {
        "gene set name": "gene_1 gene_2 gene_3 ... gene_n", 
        ...
    }
    In addition, this function augments each gene set library using ARCHS4 
    gene-gene co-expression data. For each gene set, the most co-expressed genes 
    (determined by summing the coexpression coefficients across all genes)
    are added to the gene set before visualization. 
    '''
    ### variable to store data
    raw_library_data = []

    if local: ### get library data (GMT file) locally
        with open(f'../../../Libs_to_scatter/{lib}.gmt', 'r') as f:
            for line in f.readlines():
                raw_library_data.append(line.split("\t\t"))
    else: ### get library data (GMT file) from Enrichr
        with urllib.request.urlopen('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=' + lib) as f:
            for line in f.readlines():
                raw_library_data.append(line.decode("utf-8").split("\t\t"))

    ### keep track of geneset data
    lib_dict = {}
    aug_dict = {}

    for i in range(len(raw_library_data)):
        name = raw_library_data[i][0]
        raw_genes = [gene.strip().split(',')[0] for gene in raw_library_data[i][1].split('\t')]
        if augmented:
            aug_genes = ' '.join(augment_archs4(raw_genes[:-1]))
            aug_dict[name] = aug_genes
        else:
            aug_dict[name] = None
        
        lib_dict[name] = ' '.join(raw_genes)
        
    return lib_dict, aug_dict


def str_to_int(s, i):
    '''
    Computes the sha256 hash for the input string {s}, then converts 
    it into an int mod {i}. 
    '''
    s = re.sub(r"\([^()]*\)", "", s).strip()
    byte_string = bytearray(s, "utf8")
    return int(hashlib.sha256(byte_string).hexdigest(), base=16) % i


def process(libname, libdict, data_dir, augmented=False, augdict=None):
    vec = TfidfVectorizer()
    if augmented:
        X = vec.fit_transform(augdict.values())
        adata = anndata.AnnData(X)
        adata.obs.index = augdict.keys()
    else:
        X = vec.fit_transform(libdict.values())
        adata = anndata.AnnData(X)
        adata.obs.index = libdict.keys()

    sc.pp.neighbors(adata, n_neighbors=30)
    sc.tl.leiden(adata, resolution=1.0)
    sc.tl.umap(adata, min_dist=0.1)

    new_order = adata.obs.sort_values(by='leiden').index.tolist()
    adata = adata[new_order, :]
    adata.obs['leiden'] = 'Cluster ' + adata.obs['leiden'].astype('object')

    df = pd.DataFrame(adata.obsm['X_umap'])
    df.columns = ['x', 'y']

    df['cluster'] = adata.obs['leiden'].values
    df['term'] = adata.obs.index
    df['genes'] = [libdict[l] for l in df['term']]

    df.to_csv(f"{data_dir}/{libname}.csv", index=False)


data_dir = '../../../Enrichment-Appyter/new_scatterlibs'
for (l_name, l_len) in libs:
    # if path.exists(f"{data_dir}/{l_name}.csv"): continue
    print (f"Processing {l_name}")
    if l_len < 100:
        l_dict, a_dict = get_Enrichr_library(l_name, augmented=True)
        process(l_name, l_dict, data_dir, augmented=True, augdict=a_dict)
    else:
        l_dict, _ = get_Enrichr_library(l_name)
        process(l_name, l_dict, data_dir)
    print("\tDone!")

# for lib in libs:
#     if path.exists('Libraries/' + lib + '.csv'):
#         continue
#     print("SCATTER LIB:", lib) # keep track of library

#     # handle library inaccessibility
#     try:
#         library_data = get_Enrichr_library(lib)
#     except:
#         print("failed to access", lib, "-- continuing")
#         continue

#     df = pd.DataFrame(
#         data = library_data, 
#         columns = ['Name', 'Genes', 'Augmented_Genes']
#     )

#     gene_list = df['Augmented_Genes']
    
#     print("\ttfidf") # keep track of processing step
#     try:
#         tfidf_vectorizer = TfidfVectorizer(
#             analyzer=lambda gene: gene,
#             min_df = 3,
#             max_df = 0.05,
#             max_features = 100000,
#             ngram_range=(1, 1)
#         )
#         tfidf = tfidf_vectorizer.fit_transform(gene_list)

#     except:
#         tfidf_vectorizer = TfidfVectorizer(
#             analyzer=lambda gene: gene,
#             min_df = 3,
#             max_df = 0.25,
#             max_features = 100000,
#             ngram_range=(1, 1)
#         )
#         tfidf = tfidf_vectorizer.fit_transform(gene_list)

#     print("\tumap") # keep track of processing step
#     reducer = umap.UMAP()
#     reducer.fit(tfidf)
#     embedding = pd.DataFrame(reducer.transform(tfidf), columns=['x','y'])

#     df['Genes'] = df['Genes'].apply(lambda x: ' '.join(x))
#     pd.concat([embedding, df[['Name', 'Genes']]], axis=1).to_csv('Libraries/' + lib + '.csv', index = False)

