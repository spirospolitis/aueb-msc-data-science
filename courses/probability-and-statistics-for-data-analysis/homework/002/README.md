# Export R code from .Rmd

From the R command line, execute:

## 1st method (using custom script)

### Execute the following R function

#' ## script for converting .Rmd files to .R scripts.

#' #### Kevin Keenan 2014
#' 
#' This function will read a standard R markdown source file and convert it to 
#' an R script to allow the code to be run using the "source" function.
#' 
#' The function is quite simplisting in that it reads a .Rmd file and adds 
#' comments to non-r code sections, while leaving R code without comments
#' so that the interpreter can run the commands.
#' 
#' 
rmd2rscript <- function(infile){
  # read the file
  flIn <- readLines(infile)
  # identify the start of code blocks
  cdStrt <- which(grepl(flIn, pattern = "```{r*", perl = TRUE))
  # identify the end of code blocks
  cdEnd <- sapply(cdStrt, function(x){
    preidx <- which(grepl(flIn[-(1:x)], pattern = "```", perl = TRUE))[1]
    return(preidx + x)
  })
  # define an expansion function
  # strip code block indacators
  flIn[c(cdStrt, cdEnd)] <- ""
  expFun <- function(strt, End){
    strt <- strt+1
    End <- End-1
    return(strt:End)
  }
  idx <- unlist(mapply(FUN = expFun, strt = cdStrt, End = cdEnd, 
                SIMPLIFY = FALSE))
  # add comments to all lines except code blocks
  comIdx <- 1:length(flIn)
  comIdx <- comIdx[-idx]
  for(i in comIdx){
    flIn[i] <- paste("#' ", flIn[i], sep = "")
  }
  # create an output file
  nm <- strsplit(infile, split = "\\.")[[1]][1]
  flOut <- file(paste(nm, "[rmd2r].R", sep = ""), "w")
  for(i in 1:length(flIn)){
    cat(flIn[i], "\n", file = flOut, sep = "\t")
  }
  close(flOut)
}

### Call the function

#### Linux

setwd("/home/sp/Documents/msc-in-data-science/courses/core/probability-and-statistics-for-data-analysis/homework/002/")
rmd2rscript(infile="homework-2.Rmd")

#### Windows

setwd("C:\\Users\\sp.RIRIPC\\Documents\\msc-in-data-science\\courses\\core\\probability-and-statistics-for-data-analysis\\homework\\002\\")
rmd2rscript(infile="homework-2.Rmd")

## 2nd method (using knitr)

### Linux

library(knitr)
setwd("/home/sp/Documents/msc-in-data-science/courses/core/probability-and-statistics-for-data-analysis/homework/002/")
purl("homework-2.Rmd", output = "homework-2.R", documentation = 2)

### Windows
library(knitr)
setwd("C:\\Users\\sp.RIRIPC\\Documents\\msc-in-data-science\\courses\\core\\probability-and-statistics-for-data-analysis\\homework\\002\\")
purl("homework-2.Rmd", output = "homework-2.R", documentation = 2)