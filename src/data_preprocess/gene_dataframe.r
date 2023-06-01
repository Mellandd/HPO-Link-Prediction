Genes <- read.csv("/home/mellina/tfm/src/data/mlDBGtexGenesWGHugoString.txt")
library(readr)
library(tidyverse)
ExprGenes <- Genes %>% select(gene, starts_with("Expr"))
ExACGenes <- Genes %>% select(gene, starts_with("ExAC"))
locusGenes <- Genes %>% select(gene, starts_with("Counts")) 
proteinGenes <- Genes %>% select(gene, "StringCombined")
genomicGenes <- Genes %>% select(gene, "constitutiveexons","ESTcount","alt3EST","alt5EST","alt3.5EST", "GeneLength","TranscriptCount","GCcontent","NumJunctions","IntronicLength")
allGenes <- Genes %>% select(gene, starts_with("Expr"), starts_with("ExAC"), starts_with("Counts"), "StringCombined")


url <- 'https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2022-12-15/phenotype_to_genes.txt'
dest <- '/home/mellina/tfm/src/data/phenotype_to_genes.txt'
gen_to_prot <- read.delim("~/tfm/src/data/gen_to_prot.tsv", stringsAsFactors=FALSE)

url2 <- 'https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-04-05/phenotype_to_genes.txt'

download.file(url, dest)
phenotypes_to_genes <- read.delim(dest, header=FALSE, comment.char="#")
colnames(phenotypes_to_genes)[1] <- 'HPO-id'
colnames(phenotypes_to_genes)[4] <- 'entrez-gene-symbol'
phenotypes_to_genes <- phenotypes_to_genes %>% select('HPO-id', 'entrez-gene-symbol')
phenotypes_to_genes <- unique(phenotypes_to_genes)

download.file(url2, '/home/mellina/tfm/src/data/phenotype_to_genes_new.txt')

phen_new <- read.delim("/home/mellina/tfm/src/data/phenotype_to_genes_new.txt", comment.char="#")
genes1 <- unique(ExACGenes$gene)
genes2 <- phenotypes_to_genes$`entrez-gene-symbol`
genes2 <- unique(genes2)

phen_new <- unique(phen_new)
phens <- unique(phenotypes_to_genes$`HPO-id`)
phen_new <- filter(phen_new, phen_new$hpo_id %in% phens)
colnames(phen_new)[1] <- 'HPO-id'
colnames(phen_new)[4] <- 'entrez-gene-symbol'
phen_new <- phen_new %>% select('HPO-id', 'entrez-gene-symbol')


genes <- intersect(genes1, genes2)
gen <- unique(phenotypes_to_genes$`entrez-gene-symbol`)
gen_to_prot <- filter(gen_to_prot, From %in% genes)

ExACGenes <- filter(ExACGenes, gene %in% genes)
ExprGenes <- filter(ExprGenes, gene %in% genes)
locusGenes <- filter(locusGenes, gene%in% genes)
proteinGenes <- filter(proteinGenes, gene%in% genes)
genomicGenes <- filter(genomicGenes, gene%in% genes)
allGenes <- filter(allGenes, gene %in% genes)
phenotypes_to_genes <- filter(phenotypes_to_genes, phenotypes_to_genes$`entrez-gene-symbol` %in% genes)

write.csv(ExACGenes, "/home/mellina/tfm/src/data/exACGenes.csv", row.names=FALSE)
write.csv(ExprGenes, "/home/mellina/tfm/src/data/exprGenes.csv", row.names=FALSE)
write.csv(locusGenes, "/home/mellina/tfm/src/data/ComplexityGenes.csv", row.names=FALSE)
write.csv(proteinGenes, "/home/mellina/tfm/src/data/proteinGenes.csv", row.names=FALSE)
write.csv(genomicGenes, "/home/mellina/tfm/src/data/genomicGenes.csv", row.names=FALSE)
write.csv(allGenes, "/home/mellina/tfm/src/data/allGenes.csv", row.names=FALSE)
write.csv(phenotypes_to_genes, '/home/mellina/tfm/src/data/phenotypes_to_genes.csv', row.names = FALSE)
write.csv(df, "/home/mellina/tfm/src/data/genList.csv", row.names=FALSE, quote=FALSE)

new_edges <- anti_join(phen_new, phenotypes_to_genes, by=c("entrez-gene-symbol", "HPO-id"))

rem_edges <- anti_join(phenotypes_to_genes, phen_new, by=c("entrez-gene-symbol", "HPO-id"))

agg <- gen_to_prot %>% count(From)
new <- merge(phenotypes_to_genes, agg, by.x='entrez-gene-symbol', by.y='From')
agg <- aggregate(new$n, by=list(Category=new$`HPO-id`), FUN=sum)

write.csv(agg, "/home/mellina/tfm/src/data/count_prot.csv", row.names=FALSE, quote=FALSE)

