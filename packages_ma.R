##### This script loads all packages required for the project

package_list <- c("xgboost","ranger","tcltk","data.table","tictoc","pROC","logging","magrittr","keyring","stringr","lubridate","ggplot2",
                  "PRROC","pROC","dplyr","readr","readxl","readr","readxl","stringr","tibble","tidyr","purrr","forcats","caret","e1071",
                  "Metrics","yardstick","caret","rpart","rpart.plot","scales","ComplexHeatmap","palettes","ggsci")

for (package in package_list){
  print(paste0("Loading package: ",package))
  if (!require(package,character.only = TRUE)){
    print(paste0("Package not found. Installing package: ",package))
    install.packages(package)}
  library(package,character.only = TRUE)
}
## Load functions
source("Funktionen/functions_ma.R", encoding = "UTF-8")

