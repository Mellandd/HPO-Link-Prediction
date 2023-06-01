rule metrics_gnn:
    input:
        model = "src/data/model.pt"
    shell:
        ""python src/analysis/metrics_gnn.py""


rule create_gnn:
    input:
        dataset = "src/data/dataframe.pt"
    output:
        model = "src/data/model.pt"
    shell:
        "python src/analysis/create_gnn.py"

rule create_dataset:
    input:
        phen_edges = "src/data/phenotype_edges.csv",
        phen = "src/data/phenotypes.csv",
        genes = "src/data/exACGenes.csv",
        phen_gen = "src/data/phenotypes_to_genes.csv"
    output:
        dataset = "src/data/dataframe.pt"
    shell:
        "python src/data_preprocess/create_dataframe.py"

rule create_gene_dataframe:
    output:
        genes = "src/data/exACGenes.csv",
        edges = "src/data/phenotypes_to_genes.csv"
    shell:
        "Rscript gene_dataframe.r"

rule create_graph:
    output:
        phen_edges = "src/data/phenotype_edges.csv",
        phen = "src/data/phenotypes.csv"
    shell:
        "python src/data_preprocess/download_graph.py"
