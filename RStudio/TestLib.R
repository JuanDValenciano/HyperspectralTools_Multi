# Install Lib
install.packages ("hyperSpec")

# load Lib
library(lattice)    # install.packages("ggplot2")
library(grid)    # install.packages("ggplot2")
library(ggplot2)    # install.packages("ggplot2")
library(xml2)    # install.packages("ggplot2")
library(hyperSpec)

# Clear Global Environment
rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
gc() #free up memrory and report the memory usage.

#Test2_Hyper = read.ENVI.HySpex("/home/juanval/Hyper/Experiment_2/Test/test_d2/test2_16000_us_2x_2019-11-25T124320_corr.hyspex", "/home/juanval/Hyper/Experiment_2/Test/test_d2/test2_16000_us_2x_2019-11-25T124320_corr.hdr")
Test2_Hyper = read.ENVI.HySpex("/home/juand/Hyper/Experiment_2/Tommy/control_group/sample_1/tom_cgs_01_16000_us_2x_2019-11-24T121227_corr.hyspex", "/home/juanval/Hyper/Experiment_2/Tommy/control_group/sample_1/tom_cgs_01_16000_us_2x_2019-11-24T121227_corr.hdr")
dim (Test2_Hyper)


plotspc(Test2_Hyper)
#plotmat(Test2_Hyper) # No correr
plotmap(Test2_Hyper)
