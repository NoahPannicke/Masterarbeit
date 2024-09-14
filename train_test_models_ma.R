#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Description: Configuration for the model and list creation scripts
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options(download.file.method="wininet")
Sys.setenv(NLS_LANG="German") # to enable the db connection to use umlaute
rm(list = ls()) 
gc()

memory.limit(size = 120e7)
memory.size(max = TRUE)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Path Store and Snapshot ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Erstelle Path-Store
PROD_RUN_DIR    <- paste0("prod_", format(Sys.time(), format = "%Y%m%d_%H%M%S"))

## Erstelle Snapshot-Ordner
snapshot_dir <- file.path("snapshots",
                          format(Sys.time(), format = "%Y%m%d_%H%M%S"))

## Objekte, die bei Aufruf der configs im WS behalten werden sollen
keep_list   <- c("keep_list", "config", "configs", "result",
                 "snapshot_dir", "score_date", "dt_rows_prod",
                 "PROD_RUN_DIR",'value','val_erg','val_list','i')  
## Set meta parameters
options(error=recover) #otherwise error=NULL

if(exists("keep_list")){
  
  remove_list <- ls()[!ls() %in% keep_list]
  rm(list = remove_list)
  
}

## load packages and functions
source("Funktionen/packages_ma.R", encoding = "UTF-8")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters 0: Most important parameters like Target or Stichtag ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## name of this configuration 
configuration_name <- "prodaffi_ba_score"

## Column where scores will be saved
db_column        <- "BONUSAPP_SCORE" 

## Table name to write data to
table_name       <- "PROD_BONUSAPP_SCORE"

## Comment added to the time stamp of the results folder
res_fol_comment  <- configuration_name

## Specify target: FVKU_KUENDIGUNG_6M_FLAG or FVKU_ABGANG_6M_FLAG
target           <- "V_FLSP_AOK_BONUS_AUSZ_FLAG_6M"  

## Limitation of data basis
limit            <- "FMER_BONUSAPP_STICHTAG_FLAG == '0'"

## Number of rows to use. 
## Use Inf for production.
dt_rows <- Inf

## Set reference dates: rd_test is the date for computing evaluation metrics, 
## train dates will be computed relative to rd_test based on rd_train_lag, e.g. 
## -6 means the train date is six month before test date. If more than one value
## is provided for rd_train_lag, training data will be pooled
rd_score         <- "2023-06"
rd_test          <- rd_score
rd_train         <- c(-6)

## Variable selection
## ATTENTION: If no selection is wanted, set NULL
sel_path <- "Metadaten/features/features_prodaffi_ba_top_37.csv"

## Laenge des Beobachtungszeitraums fuer die Features
beo_intervall    <- "36M"

## Aus welcher DB-Tabelle sollen die Daten geladen werden?
## NULL -> Daten werden aus SE_FEATURES_B{beo_intervall} geladen
## Wenn der Wert nicht NULL ist, muss er auf eine existierende Tabelle in der DB
## gesetzt werden
data_mode <- 'SE_FEATURES_GLOBAL'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters 1: Paths, Dates and Validation Mode ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Path for saving results
## Note: A time stamp subfolder will be created to save results
path_results     <- file.path("ergebnisse", PROD_RUN_DIR)

# save used dataset for shap calculation
save_dataset <- F

# load existing snapshot 
load_existing_snap <- T
snap_path          

## path for special training data (csv, xlsx, sql-statement)
## if not in use, set to NULL!!!
train_data_path <- NULL
test_data_path  <- NULL
score_data_path <- NULL

## Set values of evaluation criteria
## Note: AUROC will be reported by default, adding further criteria besides the
## four provided needs adjustment in the train and test loop
eval_crit_list <- list(CRate_Train    = c(0.05),
                       CRate          = c(0.01, 0.05, 0.1),
                       Lift           = c(0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
                       LiftAbs        = c(1e4, 2e4, 3e4, 4e4, 5e4, 6e4))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters 2: Features and Hyperparameters ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Target encoding  
do_te            <- T

## threshold for number of observation in order to be target encoded
target_threshold <- 1500

## Exclude highly correlated features if true
do_correlation   <- FALSE

## correlation threshold to throw out high vorrelated variables
corr_threshold   <- 0.975

## To do: check if possible to choose all factors for te.
factors_te <- c("DVER_NATION_MSA", "DVER_NATION_GR_1", "DVER_NATION_GR_2","DVER_NATION_BEZ", 
                "DARB_BRANCHE_CODE",
                "DARB_ANZ_BESCHAEFTIGTE_GES", "DARB_ANZAHL_MITGLIEDER","DARB_REGION_CODE",
                "DARB_ANZ_BESCHAEFTIGTE_GEMELD","DARB_ANZ_AZUBI_GEMELD", "DARB_ORT",
                "DARB_BUNDESLAND", 
                "FVER_VERSART_BEZ", "FVER_VARTGRUPPE_BEZ","FVER_BUNDESLAND_CODE_3", 
                "FVER_VARTHAUPTGRUPPE_BEZ",
                "DBGR_BEITRAGSGRUPPE_ID", "DPGR_ID", "DBGR_CODE",
                "DTAE_TAETIGKEIT_BEZ","FVMZ_MAX_ANMELDETEXT","FVMZ_ARBG_BEZ")

## Variablen welche immer ausgeschlossen werden sollen
always_excl <- c("DVER_GEOMARKET", 
                 "FVMZ_ENDE_DIFF_TAGE", "FVMZ_ENDE_DAT",
                 "FVMZ_FOLGEKK_BEZ", "FVMZ_FOLGEKK_DAT_MAX","DARB_BNR",
                 "FMER_K_ZEITUNG_STICHTAG_ANZ","FMER_K_ZEITUNG_ANZ",'FMER_DIGITAL_ANZ','KUENDIGERANTEIL_PLZ')

## Meta hyperparamters not depending on algorithm
hp_list_meta       <- list(sampling_ratio   = c(0), # Note: for 0 no sampling is done
                           do_weighting     = c(FALSE) 
)

## XGBoost hyperparameters
hp_list            <- list(eta               = c(0.05),
                           gamma             = c(10),
                           max_depth         = c(4),
                           #min_child_weight  = 1,
                           subsample         = c(1),
                           colsample_bytree  = c(0.3),
                           nrounds           = c(350),
                           objective         = "binary:logistic")

## Ranger hyperparameters
hp_list_ranger     <- list(num.trees         = c(100),
                           min.node.size     = c(100),
                           max.depth         = c(0),
                           mtry              = c(32)
)

## Create hyperparameter grid
hp_grid <- expand.grid(c(hp_list_meta, hp_list))

## Variables for Training
var_sel <- c("DVER_PID",
             as.character(read.csv2(sel_path)$Feature))
          
## Keep only the factors for target encoding that are in var_sel 
if(!is.null(var_sel)) factors_te <- factors_te[factors_te %in% var_sel]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters 3: Restrictions for Training and Scoring ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Define subset rules for train and test set
all_bl     <- c("BAW", "BAY", "BLN", "BRA", "BRE", "HES", "HH", "MVP", "NDS",
                "NRW", "RPF", "SAA", "SAC", "SAN", "SLH", "THÜ", "n/v")

all_bl_ger <- c("BAW", "BAY", "BLN", "BRA", "BRE", "HES", "HH", "MVP", "NDS", 
                "NRW", "RPF", "SAA", "SAC", "SAN", "SLH", "THÜ")

no_bl      <- c( "BLN", "BRA", "MVP")

au_bl      <- c("BAW", "BAY", "BRE", "HES", "HH", "NDS", 
                "NRW", "RPF", "SAA", "SAC", "SAN", "SLH", "THÜ", "n/v")

# Begrenzung auf das maximale Alter von Versicherten im Datensatz 
# um Performance im Prozess zu erhöhen
max_alter <- 70

## Note: If vartklasse is NULL the subset FVER_VERSSTATUS_BEZ %in% "Mitglieder"
train_subset_list <- list(age_min = 15, age_max = 70,
                          vartklasse = NULL, #c("AKV"), #c("Fami"),
                          bundesland = no_bl,
                          dauer_min = 0, dauer_max = Inf)

test_subset_list  <- list(age_min = 15, age_max = 70,
                          vartklasse = NULL, #c("AKV"), #c("Fami"),
                          bundesland = no_bl,
                          dauer_min = 0, dauer_max = Inf)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Print and log loading  ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

log_info("Finished loading the configuration")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Description: Train, tune and test models to predict 
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create folder and save meta data ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## Create results folder with time stamp
path_results_ts <- file.path(path_results, 
                             paste(get_timestamp(), res_fol_comment, sep = "_"))

dir.create(file.path(path_results_ts, "models"), recursive = TRUE)
dir.create(file.path(path_results_ts, "predictions"), recursive = TRUE)
dir.create(file.path(path_results_ts, "targetmapping"), recursive = TRUE)
dir.create(file.path(path_results_ts, 'rules'), recursive = TRUE)

# save relevant objects for use in scoring script but remove objects greater 
# than 10mb to save time. All relevant objects are only variables or small lists
# and should be smaller than 10mb.
size_in_mb <- sapply(ls(envir=globalenv()), 
                     function(x) object.size(get(x)))/(1024^2)

# PATH_STORE will never be saved because it is only relevant in prod mode
save(list = ls()[ls() %nin% c("PATH_STORE", size_in_mb[size_in_mb > 10])],
     file = paste0(path_results_ts, "/used_workspace.RData"))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Snapshots ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Initialisiere Liste und weise den Teststichtagen die Trainingsstichtage zu
test_train_list         <- vector(mode = "list", length = length(rd_test))

# Renaming of rd_test if alternative data is used
if(!is.null(test_data_path)){
  
  rd_test_man             <- strsplit(test_data_path,'/')[[1]][length(strsplit(test_data_path,'/')[[1]])]
  names(test_train_list)  <- rd_test
  
}else{
  
  names(test_train_list)  <- rd_test
  
}

# Anpassung der ST Formate 
for(rd_iter in rd_test){
  
  # Umformung von z.B. -6 in Datum
  rd_iter_ymd  <- add_month(as.Date(paste0(rd_iter, "-01")) - 1, 1)
  
  rd_train_ymd <- do.call(c, lapply(rd_train, add_month, start = rd_iter_ymd))
  rd_train_ym  <- substr(rd_train_ymd, start = 1, stop = 7)
  
  rd_train_lag <- rd_train_ym
  
  # Bedingung für manuelle Datensätze
  if(!is.null(train_data_path)){
    
    test_train_list[[rd_iter]][["train_dates"]] <- strsplit(train_data_path,'/')[[1]][length(strsplit(train_data_path,'/')[[1]])]
    
  }else{
    
    test_train_list[[rd_iter]][["train_dates"]] <- rd_train_ym
    test_train_list[[rd_iter]][["score_dates"]] <- rd_score
    
  }
}

# rename of list if manual dataset is used
if(!is.null(test_data_path)){names(test_train_list)[names(test_train_list) == rd_test] <- rd_test_man}

# all test data base files
rd_to_load   <- unique(c(rd_test, unlist(test_train_list)))

# check if snapshots are already in snapshot directory
# Namen der Tabelle in der DB erstellen
db_tab_name      <- ifelse(is.null(data_mode),
                           paste0("SE_FEATURES_B", beo_intervall),
                           data_mode)

snapshot_pattern <- tolower(db_tab_name)

# Welche Snapshots existieren bereits unter snapshot_dir? Diese muessen nicht aus
# der DB geladen werden.
snapshot_dir <- file.path("snapshots",
                          format(Sys.time(), format = "%Y%m%d_%H%M%S"))

snapshot_exist <- sapply(rd_to_load, 
                         function(x) any(grepl(pattern = paste0(snapshot_pattern, "_", x),
                                               list.files(snapshot_dir))))

if(load_existing_snap == F) dir.create(snapshot_dir)

if(any(snapshot_exist)){
  
  
  message(paste0("snapshot directory has already files with pattern ",
                 paste0(names(snapshot_exist[snapshot_exist]), collapse = ", ")))
  
  message(paste0("snapshots to load: ",
                 ifelse(all(snapshot_exist), 
                        "none",
                        paste0(names(snapshot_exist[!snapshot_exist]), collapse = ", "))))
  
}

## Datenabfrage 
if(load_existing_snap == FALSE){
  
  grp_datasets <- list('train_data_path'=train_data_path,
                       'test_data_path' =test_data_path,
                       'score_data_path'=score_data_path)
  
  erg_datasets <- list('train_dataset'=NULL,
                       'test_dataset' =NULL,
                       'score_dataset'=NULL)
  
  for(i in 1:length(grp_datasets)){
    
    if(!is.null(grp_datasets[[i]])){
      
      if(grepl('.csv',grp_datasets[[i]])){
        
        print('CSV-Datei eingeladen.')
        erg_datasets[[i]] <- fread(grp_datasets[[i]],stringsAsFactors = F)
        
      }else if(grepl('.xlsx',grp_datasets[[i]])){
        
        print('XLSX-Datei eingeladen.')
        erg_datasets[[i]] <- read_xlsx(grp_datasets[[i]])
        
      }else{
        
        print('DB-Datei eingeladen.')
        erg_datasets[[i]] <- get_data(grp_datasets[[i]])
        
      }
    }
  }
  
  if(is.null(erg_datasets[[3]])){
    
    if(rd_test == rd_score & is.null(erg_datasets[[2]])){
      
      print('Score Stichtag bereits beim Test Stichtag heruntergeladen!')
      
    }else{
      
      create_snapshots2(time_seq     = rd_score,
                        tab_name     = db_tab_name,
                        snapshot_dir = snapshot_dir,
                        selection    = sel_path)
    }
    
  }
  
  if(is.null(erg_datasets[[2]])){
    
    create_snapshots2(time_seq     = rd_iter,
                      tab_name     = db_tab_name,
                      snapshot_dir = snapshot_dir,
                      selection    = sel_path)
    
  }
  
  if(is.null(erg_datasets[[1]])){
    
    if(grepl('^-.*',rd_train_lag)){
      rd_train_lag <- sapply(rd_train_lag,
                             function(x)
                             {substr(ym(rd_test) %m-% months(x*(-1)),1,7)})
    }
    
    create_snapshots2(time_seq     = rd_train_lag, 
                      tab_name     = db_tab_name,
                      snapshot_dir = snapshot_dir,
                      selection    = sel_path)
    
  }
  
}else{
  
  erg_datasets <- list('train_dataset'=NULL,
                       'test_dataset' =NULL,
                       "score_dataset"=NULL)
  
  snapshot_dir <- snap_path
  
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training and tuning ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Initialize data frame to save performance results and folder to temporarily
## save results after each iteration
lift_df       <- data.frame()
lift_tmp_path <- file.path(path_results_ts, "tmp")
dir.create(lift_tmp_path)

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Prepare data
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rd_train_iter <- test_train_list[[rd_test]][["train_dates"]]
rd_test_iter  <- names(test_train_list)

rd_score_iter <- test_train_list[[rd_test]][["score_dates"]]

if(is.null(erg_datasets[[1]])){
  
  trained_list    <- load_prepared_data()
  
}else{
  
  trained_list    <- load_prepared_train_alternative_data(alternative_data = erg_datasets)
  
}

train_data_iter <- trained_list$train_data_iter
test_data_iter  <- trained_list$test_data_iter
score_data_iter <- trained_list$score_data_iter
score_data_st   <- trained_list$score_data_st

high_corr_names <- trained_list$high_corr_names
target_enc_list <- trained_list$target_enc_list

gc()

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Iterate over all hyperparameter tuples
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Determine number of rows in grid for lapply
grid_rows  <- nrow(hp_grid)

model_list <- list()

for(i in 1:5){
  
lift_df       <- data.frame()

for(x in seq_len(grid_rows)){
  
  print(paste0('HPO-Kombination: ',x))
  gc()
  
  ## Determine name of file to save model 
  model_name <- paste0("rd_", gsub("-", "", max(rd_train_iter)), 
                       "_iter_", add_leading_zeros(x, nchar(nrow(hp_grid))), ".models")
  
  model_file <- file.path(path_results_ts, "models", model_name)
  
  ## Get Sampling IDs. Note: Sampling  is deliberately done within the sapply-
  ## statement since it induces some randomness to the algorithm and is 
  ## therefore in scope of the n_runs-purpose.
  sampling_ids   <- get_sampling_ids(target = train_data_iter$TARGET,
                                     neg2pos_ratio = hp_grid$sampling_ratio[x])
  
  ## Train Model
  train_features <- train_data_iter[sampling_ids, 
                                    names(train_data_iter) %nin% c("TARGET", "META_STICHTAG", "DVER_PID"),  
                                    with = FALSE]
  
  
  train_par_list <- as.list(hp_grid[x, ][, !(names(hp_grid[x, ]) %in% c("nrounds", 
                                                                        "sampling_ratio",
                                                                        "do_weighting"))])
  
  rows_total  <- nrow(train_features)
  shuffle_idx <- sample(1:rows_total,size = rows_total, replace = FALSE)
  #shuffle_test_idx <- sample(1:rows_total,size = nrow(test_data_iter), replace = FALSE)
  
  train_idx   <- shuffle_idx[1:round(0.8*rows_total, 0)]
  test_idx    <- shuffle_idx[(round(0.8*rows_total, 0)+1):rows_total]
  
  train_data_xgb <- xgboost::xgb.DMatrix(data   = data.matrix(train_features),
                                         label  = train_data_iter$TARGET)
  
  test_data_xgb  <- xgboost::xgb.DMatrix(data  = data.matrix(test_data_iter[,names(test_data_iter) %nin% c("TARGET",
                                                                                                           "META_STICHTAG", 
                                                                                                           "DVER_PID"), with = FALSE]),
                                         label = test_data_iter$TARGET)

  wl_xgb         <- xgboost::xgb.DMatrix(data   = data.matrix(train_features[test_idx]),
                                         label  = train_data_iter$TARGET[test_idx])
  
  # Model mit xgboost
  boost_model <- xgboost::xgb.train(params  = train_par_list,
                                    data    = train_data_xgb,
                                    nrounds = hp_grid[x, ]$nrounds,
                                    eval_metric = "auc",
                                    watchlist = list(train = train_data_xgb,
                                                     eval = wl_xgb),
                                    callbacks = list(cb.early.stop = cb.early.stop(stopping_rounds = 40,
                                                                                   maximize = TRUE,
                                                                                   metric_name = 'eval-auc',
                                                                                   verbose = TRUE)),
   
                                   verbose = 1)
  
  # Model mit rpart
  #boost_model <- rpart::rpart(formula = TARGET ~ ., data = train_data_xgb, method = "class")
  
  # Model mit e1071
  # sample_svm <- sample_n(train_data_iter, 150000)
  # pos_tar    <- train_data_iter[TARGET == 1,]
  # sample_svm <- rbind(sample_svm, pos_tar)
  # 
  # boost_model <- e1071::svm(formula = TARGET ~ ., data = sample_svm, method = "class")
  
  pred_test       <- predict(boost_model,
                             test_data_xgb,
                             ntreelimit = boost_model$best_iteration,
                             type = "prob")
  
  temp <- calc_eval_metrics(pred_test = pred_test,
                    y_test    = test_data_iter$TARGET,
                    eval_crit_list = eval_crit_list)
  print(temp)
  lift_df <- rbind(lift_df, temp)
  }
}

## Alternative Evaluationsmetriken----

xtab <- table(ifelse(pred_test < 0.025,0,1),test_data_iter$TARGET)

caret::confusionMatrix(xtab, positive = "1")

Metrics::precision(ifelse(pred_test < 0.025,0,1),test_data_iter$TARGET)
Metrics::recall(ifelse(pred_test < 0.025,0,1),test_data_iter$TARGET)

# Recall-Precision curve   
RP.perf <- performance(pred_test, "prec", "rec")

plot(RP.perf)

plot(performance(pred_test, "f"))

# ROC curve
ROC.perf <- performance(pred_test, "tpr", "fpr")
plot(ROC.perf)

# ROC area under the curve
auc.tmp <- performance(pred_test,"auc")
auc <- as.numeric(auc.tmp@y.values)

all_erg <- data.frame(pred = pred_test, obs = test_data_iter$TARGET)
all_erg <- all_erg[order(-all_erg$pred),]
setDT(all_erg)
all_erg <- all_erg[1:9700,]
table(all_erg$obs)

## Tatsächliche Produduktnutzer----
# Liniendiagramm erstellen
ggplot(df, aes(x = quant, y = Tatsaechliche_Produktnutzung)) +
geom_line(group=1, colour="#E69F00") +  # Linie in spezifischer Farbe zeichnen
geom_point(color = "#E69F00", size = 3) +  # Punkte in spezifischer Farbe
labs(title = "Produktnutzung nach Quantilen",
     x = "Quantile",
     y = "Produktnutzung in Prozent") +
scale_x_continuous(breaks = seq(1, 20, by = 1)) + # X-Achse Anpassung
scale_y_continuous(labels = scales::percent_format(scale = 100),  # Y-Achse als Prozent
                   limits = c(0, max(df$Tatsaechliche_Produktnutzung))) + # Setzt die Grenzen von Y-Achse
theme_minimal()  # Minimalistisches Thema für bessere Lesbarkeit

## Lift-Rest Analyse----
library(scales)
data <- data.frame(
Lift = factor(c("1%", "5%", "10%", "15%", "20%", "25%", "30%"),
              levels = c("1%", "5%", "10%", "15%", "20%", "25%", "30%")),
Percent = 100-c(9.1, 28.3, 45.1, 56.9, 66.4, 72.7, 77.9),
Percent_Reg = c(94.7, 76, 59.4, 46.3, 36.5, 29.2, 23.5)# Prozentwerte
)

ggplot(data, aes(x = Lift, y = Percent)) +
geom_line(group = 1, color = "#E69F00") +  # Linie in spezifischem Orange zeichnen
geom_point(color = "#E69F00", size = 3) +  # Punkte in spezifischem Orange
labs(title = "Nicht erfasste positive Zielvariablenausprägungen",
     x = "Anteil der absteigend sortierten Zielwahrscheinlichkeiten",
     y = "Anteil der nicht erfassten positiven Ausprägungen") +
scale_y_continuous(
  limits = c(0, 100), 
  breaks = seq(0, 100, by = 10),
  labels = percent_format(scale = 1)  # Nutzt scales library für Prozentformatierung
) +
theme_minimal() +  # Minimalistisches Thema
theme(legend.position = "none")  # Legende entfernen

ggplot(data) +
geom_line(aes(x = Lift, y = Percent, group = 1, color = "Aktive Produktnutzende"), size = 1) +
geom_point(aes(x = Lift, y = Percent, color = "Aktive Produktnutzende"), size = 3) +
geom_line(aes(x = Lift, y = Percent_Reg, group = 1, color = "Registrierungen"), size = 1) +
geom_point(aes(x = Lift, y = Percent_Reg, color = "Registrierungen"), size = 3) +
labs(title = "Nicht erfasste positive Zielvariablenausprägungen",
     x = "Anteil der absteigend sortierten Zielwahrscheinlichkeiten",
     y = "Anteil der nicht erfassten positiven Ausprägungen",
     color = "Legende") +  # Setzt die Überschrift der Legende
scale_y_continuous(
  limits = c(0, 100),
  breaks = seq(0, 100, by = 10),
  labels = percent_format(scale = 1)
) +
scale_color_manual(values = c("Aktive Produktnutzende" = "#E69F00", "Registrierungen" = "grey"),
                   labels = c("Aktive Produktnutzende" = "Aktive Produktnutzende", "Registrierungen" = "Registrierungen")) +
theme_minimal() +
theme(legend.position = "right")  # Positioniert die Legende rechts) 

## HPO-Analyse----
data_colsample <- data.table(
  colsample_bytree = c(0.7, 0.3, 0.5),
  mean_auc = c(0.8151109, 0.8145307, 0.8153470),
  mean_lift_1 = c(8.821677, 9.121059, 8.993999),
  mean_lift_5 = c(5.641019, 5.646581, 5.644727),
  mean_lift_10 = c(4.355955, 4.322528, 4.347721)
)

data_eta <- data.table(
  eta = c(0.05, 0.03, 0.07, 0.10),
  mean_auc = c(0.8158928, 0.8126883, 0.8161713, 0.8152324),
  mean_lift_1 = c(9.104699, 8.838582, 9.050167, 8.922198),
  mean_lift_5 = c(5.664285, 5.613972, 5.663558, 5.634621),
  mean_lift_10 = c(4.354155, 4.299116, 4.369206, 4.345794)
)

# Tabelle für colsample_bytree
kable(data_colsample, format = "html", caption = "Durchschnittswerte der Metriken nach colsample_bytree") %>%
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = F) %>%
  column_spec(1, bold = TRUE, border_right = TRUE) %>%
  column_spec(2:5, width = "3cm")

# Tabelle für eta
kable(data_eta, format = "html", caption = "Durchschnittswerte der Metriken nach eta") %>%
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = F) %>%
  column_spec(1, bold = TRUE, border_right = TRUE) %>%
  column_spec(2:5, width = "3cm")


lift_ord[,.(mean_auc = mean(AUROC), mean_lift_1 = mean(Lift_1), mean_lift_5 = mean(Lift_5), mean_lift_10 = mean(Lift_10)),colsample_bytree]
lift_ord[,.(mean_auc = mean(AUROC), mean_lift_1 = mean(Lift_1), mean_lift_5 = mean(Lift_5), mean_lift_10 = mean(Lift_10)),max_depth]

# Daten für die Heatmap vorbereiten
# Pivotieren der Daten für die Heatmap
eval_met <- 'Lift_10'

data_pivot <- dcast(lift_hpo, eta + gamma + colsample_bytree ~ max_depth + nrounds, value.var = eval_met)
setDT(data_pivot)

library(ComplexHeatmap)

# Erstellung einer Matrix aus der Pivot-Tabelle
heatmap_matrix <- as.matrix(data_pivot)
rownames(heatmap_matrix) <- apply(data_pivot[, .(eta, gamma, colsample_bytree)], 1, paste, collapse = "-")
heatmap_matrix <- heatmap_matrix[,-c(1:3)]

# Heatmap zeichnen
Heatmap(heatmap_matrix,
        name = eval_met,
        row_title = "Eta-Gamma-ColsampleByTree",
        column_title = "MaxDepth-Nrounds",
        clustering_distance_rows = "euclidean",
        clustering_distance_columns = "euclidean",
        show_row_names = TRUE,
        show_column_names = TRUE)
# e-g-c_m-n_Lift_10
######################################
data_pivot <- dcast(lift_hpo, max_depth + gamma + colsample_bytree ~ eta + nrounds, value.var = eval_met)
setDT(data_pivot)

library(ComplexHeatmap)

# Erstellung einer Matrix aus der Pivot-Tabelle
heatmap_matrix <- as.matrix(data_pivot)
rownames(heatmap_matrix) <- apply(data_pivot[, .(max_depth, gamma, colsample_bytree)], 1, paste, collapse = "-")
heatmap_matrix <- heatmap_matrix[,-c(1:3)]

# Heatmap zeichnen
Heatmap(heatmap_matrix,
        name = eval_met,
        row_title = "MaxDepth-Gamma-ColsampleByTree",
        column_title = "Eta-Nrounds",
        clustering_distance_rows = "euclidean",
        clustering_distance_columns = "euclidean",
        show_row_names = TRUE,
        show_column_names = TRUE)
# m-g-c_e-n_lift_10
######################################
data_pivot <- dcast(lift_hpo, nrounds + gamma + colsample_bytree ~ eta + max_depth, value.var = eval_met)
setDT(data_pivot)

library(ComplexHeatmap)

# Erstellung einer Matrix aus der Pivot-Tabelle
heatmap_matrix <- as.matrix(data_pivot)
rownames(heatmap_matrix) <- apply(data_pivot[, .(nrounds, gamma, colsample_bytree)], 1, paste, collapse = "-")
heatmap_matrix <- heatmap_matrix[,-c(1:3)]

# Heatmap zeichnen
Heatmap(heatmap_matrix,
        name = eval_met,
        row_title = "Nrounds-Gamma-ColsampleByTree",
        column_title = "Eta-MaxDepth",
        clustering_distance_rows = "euclidean",
        clustering_distance_columns = "euclidean",
        show_row_names = TRUE,
        show_column_names = TRUE)
# n-g-c_e-m_Lift_10

## HPO-Validierung----
juni <- c(
  "zwischenspeicher_ma/hpo_06_23_1.csv",
  "zwischenspeicher_ma/hpo_06_23_2.csv",
  "zwischenspeicher_ma/hpo_06_23_3.csv",
  "zwischenspeicher_ma/hpo_06_23_4.csv",
  "zwischenspeicher_ma/hpo_06_23_5.csv"
)

juni_22 <- c(
  "zwischenspeicher_ma/hpo_06_22_1.csv",
  "zwischenspeicher_ma/hpo_06_22_2.csv",
  "zwischenspeicher_ma/hpo_06_22_3.csv",
  "zwischenspeicher_ma/hpo_06_22_4.csv",
  "zwischenspeicher_ma/hpo_06_22_5.csv"
)

dez <- c(
  "zwischenspeicher_ma/hpo_12_22_1.csv",
  "zwischenspeicher_ma/hpo_12_22_2.csv",
  "zwischenspeicher_ma/hpo_12_22_3.csv",
  "zwischenspeicher_ma/hpo_12_22_4.csv",
  "zwischenspeicher_ma/hpo_12_22_5.csv"
)

# einladen und rbind
eval_met <- 'Lift_5'

juni_df <- rbindlist(lapply(juni, fread))
juni_22_df <- rbindlist(lapply(juni_22, fread))
dez_df <- rbindlist(lapply(dez, fread))

juni_df <- cbind(juni_df, hp_grid)
juni_22_df <- cbind(juni_22_df, hp_grid)
dez_df <- cbind(dez_df, hp_grid)

all_df <- rbind(juni_df, juni_22_df, dez_df)

all_df <- all_df[,.(AUROC,Lift_1,Lift_5,Lift_10,eta,gamma,max_depth,colsample_bytree,nrounds)]

data_pivot <- dcast(all_df, eta + gamma + colsample_bytree ~ max_depth + nrounds, value.var = eval_met, fun.aggregate = mean)
setDT(data_pivot)

# Erstellung einer Matrix aus der Pivot-Tabelle
heatmap_matrix <- as.matrix(data_pivot)
rownames(heatmap_matrix) <- apply(data_pivot[, .(eta, gamma, colsample_bytree)], 1, paste, collapse = "-")
heatmap_matrix <- heatmap_matrix[,-c(1:3)]

# Heatmap zeichnen
Heatmap(heatmap_matrix,
        name = eval_met,
        row_title = "Eta-Gamma-ColsampleByTree",
        column_title = "MaxDepth-Nrounds",
        clustering_distance_rows = "euclidean",
        clustering_distance_columns = "euclidean",
        show_row_names = TRUE,
        show_column_names = TRUE)

# Mittelwerte für jede Metrik und Hyperparameterkombination berechnen
summary_data <- all_df[, .(
  mean_AUROC = mean(AUROC, na.rm = TRUE),
  mean_Lift1 = mean(Lift_1, na.rm = TRUE),
  mean_Lift10 = mean(Lift_10, na.rm = TRUE),
  mean_Lift5 = mean(Lift_5, na.rm = TRUE)
), by = .(eta, gamma, max_depth, colsample_bytree, nrounds)]

# Die besten Werte für jede Metrik finden
best_AUROC <- summary_data[which.max(mean_AUROC)]
best_Lift1 <- summary_data[which.max(mean_Lift1)]
best_Lift5 <- summary_data[which.max(mean_Lift5)]
best_Lift10 <- summary_data[which.max(mean_Lift10)]

# Ausgabe der besten Kombinationen
print("Beste Kombination für AUROC:")
print(best_AUROC)
print("Beste Kombination für Lift 1:")
print(best_Lift1)
print("Beste Kombination für Lift 10:")
print(best_Lift10)

# Ermitteln der besten gemeinsamen Kombination
common_best <- summary_data[
  mean_AUROC >= 0.98 * best_AUROC$mean_AUROC &
    mean_Lift1 >= 0.98 * best_Lift1$mean_Lift1 &
    mean_Lift10 >= 0.98 * best_Lift10$mean_Lift10 &
    mean_Lift10 >= 0.98 * best_Lift5$mean_Lift5
]

common_best <- common_best[max_depth != '6' & gamma != '5',]
# Ausgabe der besten gemeinsamen Kombinationen
print("Beste gemeinsame Kombinationen:")
print(common_best)

juni_df[,.(mean_auc = mean(AUROC), mean_lift_1 = mean(Lift_1), mean_lift_5 = mean(Lift_5), mean_lift_10 = mean(Lift_10)),eta]
juni_df[,.(mean_auc = mean(AUROC), mean_lift_1 = mean(Lift_1), mean_lift_5 = mean(Lift_5), mean_lift_10 = mean(Lift_10)),max_depth]
juni_df[,.(mean_auc = mean(AUROC), mean_lift_1 = mean(Lift_1), mean_lift_5 = mean(Lift_5), mean_lift_10 = mean(Lift_10)),colsample_bytree]
juni_df[,.(mean_auc = mean(AUROC), mean_lift_1 = mean(Lift_1), mean_lift_5 = mean(Lift_5), mean_lift_10 = mean(Lift_10)),nrounds]

data <- data.frame(
  eta = c(0.05, 0.07, 0.05, 0.07),
  gamma = c(10, 10, 10, 10),
  max_depth = c(4, 4, 4, 4),
  colsample_bytree = c(0.3, 0.3, 0.3, 0.5),
  nrounds = c(300, 400, 400, 400),
  mean_AUROC = c(0.8208227, 0.8196486, 0.8204173, 0.8191346),
  mean_Lift1 = c(9.317715, 9.227342, 9.262944, 9.255660),
  mean_Lift10 = c(4.465856, 4.438571, 4.459729, 4.427328)
)

## Leistungsvergleich Algorithmen ----
# Daten laden
data <- data.frame(
  Algorithmus = rep(c("SVM", "XGB", "RF","NN"), each = 2),
  Metric = rep(c("AUC", "Top-1%-Lift"), 4),
  Value = c(0.63, 4.18, 0.78182832, 7.29387586, 0.762, 6.41, 0.7961, 5.38)
)

# Daten für ggplot vorbereiten
data_long <- data %>%
  mutate(Type = ifelse(Metric == "AUC", "AUC", "Lift"))

# Daten anzeigen
print(data_long)

# Basisplot
ggplot(data_long, aes(x = Algorithmus, y = Value, fill = Algorithmus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~Metric, scales = "free_y") +
  theme_minimal() +
  labs(title = "Leistungsvergleich von Modellalgorithmen anhand der AUC und des Top-1%-Lifts", x = "Algorithmus", y = "Wert") +
  scale_fill_uchicago('dark')


data_long <- data %>%
  mutate(Type = ifelse(Metric == "AUC", "AUC", "Lift"))
# Plot für AUC
plot_auc <- ggplot(data_long %>% filter(Metric == "AUC"), aes(x = Algorithmus, y = Value, fill = Algorithmus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "Vergleich der AUC Werte", x = "Algorithmus", y = "AUC") +
  scale_fill_uchicago('dark')

# Plot für Top-1%-Lift
plot_lift <- ggplot(data_long %>% filter(Metric == "Top-1%-Lift"), aes(x = Algorithmus, y = Value, fill = Algorithmus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  theme_minimal() +
  labs(title = "Vergleich der Top-1%-Lift Werte", x = "Algorithmus", y = "Top-1%-Lift") +
  scale_fill_uchicago('dark')
grid.arrange(plot_auc, plot_lift, ncol = 2)

# Lift_10_5
data <- data.frame(
  Algorithmus = rep(c("SVM", "XGB", "RF", "NN"), each = 2),
  Metric = rep(c("Top-5%-Lift", "Top-10%-Lift"), 4),
  Value = c(3.15, 2.64, 4.81, 3.92, 4.69, 3.69, 4.84, 4.19)
)

# Daten für ggplot
data_long <- data %>%
  mutate(Type = ifelse(Metric == "Top-5%-Lift", "Top-5%-Lift", "Top-10%-Lift")) %>%
  mutate(Metric = factor(Metric, levels = c("Top-5%-Lift", "Top-10%-Lift")))  # Ensure the order of Metric

# Display the data
print(data_long)

# Base plot
p <- ggplot(data_long, aes(x = Algorithmus, y = Value, fill = Algorithmus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~Metric, scales = "free_y") +
  theme_minimal() +
  labs(title = "Leistungsvergleich von Modellalgorithmen anhand des Top-5%- und Top-10%-Lifts", x = "Algorithmus", y = "Wert") +
  scale_fill_uchicago('dark')

data_long <- data %>%
  mutate(Type = ifelse(Metric ==  "Top-5%-Lift", "Top-5%-Lift", "Top-10%-Lift"))
# Plot für AUC
plot_auc <- ggplot(data_long %>% filter(Metric == "Top-5%-Lift"), aes(x = Algorithmus, y = Value, fill = Algorithmus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "Vergleich der Top-5%-Lift Werte", x = "Algorithmus", y = "Top-5%-Lift") +
  scale_fill_uchicago('dark')

# Plot für Top-1%-Lift
plot_lift <- ggplot(data_long %>% filter(Metric == "Top-10%-Lift"), aes(x = Algorithmus, y = Value, fill = Algorithmus)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  theme_minimal() +
  labs(title = "Vergleich der Top-10%-Lift Werte", x = "Algorithmus", y = "Top-10%-Lift") +
  scale_fill_uchicago('dark')
grid.arrange(plot_auc, plot_lift, ncol = 2)

## Übereinstimmung Algos----
xgb <- read.csv("zwischenspeicher_ma/xgb_pred.csv")
svm <- read.csv("zwischenspeicher_ma/svm_pred.csv")
nn <- read.csv("zwischenspeicher_ma/nn_pred.csv")
rf <- read.csv("zwischenspeicher_ma/rf_pred.csv")

xgb <- xgb[order(xgb$pred,decreasing = T),]
svm <- svm[order(svm$pred,decreasing = T),]
nn <- nn[order(nn$pred,decreasing = T),]
rf <- rf[order(rf$pred,decreasing = T),]


top_perc <- c(0.01,0.1)
erg <- list()

for(i in 1:length(top_perc)){
  
  xgb_top <- xgb[1:round(nrow(xgb)*top_perc[i]),]
  svm_top <- svm[1:round(nrow(svm)*top_perc[i]),]
  nn_top <- nn[1:round(nrow(nn)*top_perc[i]),]
  rf_top <- rf[1:round(nrow(rf)*top_perc[i]),]
  
  xgb_svm_common <- (length(intersect(xgb_top$dver, svm_top$dver))/nrow(xgb[1:round(nrow(xgb)*top_perc[i]),]))*100
  xgb_nn_common <- (length(intersect(xgb_top$dver, nn_top$dver))/nrow(xgb[1:round(nrow(xgb)*top_perc[i]),]))*100
  xgb_rf_common <- (length(intersect(xgb_top$dver, rf_top$dver))/nrow(xgb[1:round(nrow(xgb)*top_perc[i]),]))*100
  svm_nn_common <- (length(intersect(svm_top$dver, nn_top$dver))/nrow(xgb[1:round(nrow(xgb)*top_perc[i]),]))*100
  svm_rf_common <- (length(intersect(svm_top$dver, rf_top$dver))/nrow(xgb[1:round(nrow(xgb)*top_perc[i]),]))*100
  nn_rf_common <- (length(intersect(nn_top$dver, rf_top$dver))/nrow(xgb[1:round(nrow(xgb)*top_perc[i]),]))*100
  
  # Erstellen des Barplots
  barplot_data <- data.frame(
    Modellpaare = c("XGB-SVM", "XGB-NN", "XGB-RF", "SVM-NN", "SVM-RF", "NN-RF"),
    Common = c(xgb_svm_common, xgb_nn_common, xgb_rf_common, svm_nn_common, svm_rf_common, nn_rf_common)
  )
  
  p <- ggplot(barplot_data, aes(x = Modellpaare, y = Common, fill = Modellpaare)) +
    geom_bar(stat = "identity") +
    labs(title = paste0("Anteil der Überschneidungen in den Top ", top_perc[i]*100,"% der Vorhersagen"), y = "Prozentsatz", x = "Modellpaare") +
    scale_y_continuous(limits = c(0, 100))+
    scale_fill_uchicago('dark')+
    theme(legend.position = 'none')
  theme_minimal()
  
  erg[[i]] <- p
}

grid.arrange(erg[[1]], erg[[2]], ncol = 1)

test_data_iter[FMER_MEINEAOK_ANZ > 1,FMER_MEINEAOK_ANZ := FMER_MEINEAOK_ANZ == 1]
table(test_data_iter$FMER_MEINEAOK_ANZ
)
mean(test_data_iter[DVER_PID %in% xgb_top$dver,DGEO_KFZ_GES_RAT])
mean(test_data_iter[DVER_PID %in% svm_top$dver,DGEO_KFZ_GES_RAT])
mean(test_data_iter[DVER_PID %in% nn_top$dver,DGEO_KFZ_GES_RAT])
mean(test_data_iter[DVER_PID %in% rf_top$dver,DGEO_KFZ_GES_RAT])


test_data_iter[DARB_ORT != 'Berlin', DARB_ORT := 'Nicht Berlin']
table(test_data_iter[DVER_PID %in% xgb_top$dver,DGEO_KFZ_GES_RAT])
table(test_data_iter[DVER_PID %in% svm_top$dver,DGEO_KFZ_GES_RAT])
table(test_data_iter[DVER_PID %in% nn_top$dver,DGEO_KFZ_GES_RAT])
table(test_data_iter[DVER_PID %in% rf_top$dver,DGEO_KFZ_GES_RAT])

## Anteil -1 in den Daten----
summary(train_data_iter)
sub <- c('FFFD_SKELETT_MUSKEL_DIFF_TAGE','FFAL_LETZTE_KH_TAGE','FAUD_LETZTE_DIAG_TAGE','FFFD_SCHWANGER_DIFF_TAGE','FZBF_ZAHN_BES_BETRAG_SUM','CRM_LK_DIFF_TAGE','FFFA_LETZTE_AMB_BEH','FZBF_ZAHN_BES_ANZ')

proportions_data <- sapply(train_data_iter[,..sub], function(x) {
  sum(x == -1, na.rm = TRUE) / sum(!is.na(x)) * 100  # Prozentsatz der -1 Werte
})
proportions_data <- data.frame(Variable = names(proportions_data), Proportion = proportions_data)
proportions_data <- proportions_data[order(-proportions_data$Proportion), ]
proportions_data$Variable <- factor(proportions_data$Variable, levels = proportions_data$Variable)


ggplot(proportions_data, aes(x = Variable, y = Proportion)) +
  geom_col(fill = "#E59F01") +  # geom_col ist identisch zu geom_bar(stat="identity")
  theme_minimal() +
  labs(title = "Anteil von -1 Werten in verschiedenen Variablen",
       x = "Variable",
       y = "Prozentsatz") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## TE-Treshhold und Merkmalsauswahl----
# Sample sizes and corresponding accuracy scores
sample_sizes <- c(500, 1000, 2000, 4000, 8000, 12000, 20000)
accuracy_scores <- c(0.7940627, 0.7954411, 0.7940714, 0.7932368, 0.7880397, 0.7867073, 0.7874189)

# Creating the plot
# save plot
png("Vorstellung/TE_tresh.png",width = 600)
plot(sample_sizes, accuracy_scores, type = "b", pch = 19, col = "blue", xlab = "Schwellenwert für Restgruppe", ylab = "AUC",
     main = "Modellgenauigkeit in Abhängigkeit des Target Encoding Schwellenwerts")
grid()
dev.off()

# Daten erstellen
data <- data.frame(
  AnzahlVariablen = c(500, 400, 350, 300, 250, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10),
  AUROC = c(0.81663465, 0.8187635, 0.8167588, 0.81896862, 0.8152874, 0.81701926, 0.81878734, 0.81852655, 0.81864948, 0.818893636, 0.817184689, 0.816562434, 0.818402412, 0.815888757, 0.81739247, 0.817239818, 0.816172092, 0.816862263, 0.816092749, 0.817313036, 0.817395917, 0.815210958, 0.815310616, 0.814126909, 0.81392008, 0.813124157, 0.81416448, 0.812544824, 0.805259287),
  Lift_1 = c(8.83421946, 8.95200905, 8.71642986, 9.06979864, 9.2765362, 8.83421946, 9.01090385, 9.29985181, 9.128693439, 9.010903847, 8.716429865, 9.010903847, 8.95200905, 8.893114254, 9.069798643, 9.069798643, 9.128693439, 8.539745475, 8.245271494, 8.775324661, 8.95200905, 8.834219457, 8.775324661, 8.716429865, 8.716429865, 8.775324661, 8.480850679, 8.716429865, 8.539745475),
  Lift_5 = c(5.77145469, 5.74789773, 5.7125623, 5.73611926, 5.81856861, 5.58299903, 5.78323317, 5.83034709, 5.818568608, 5.830347087, 5.606555987, 5.689005339, 5.806790129, 5.736119255, 5.689005339, 5.665448381, 5.465214239, 5.606555987, 5.370986407, 5.712562297, 5.830347087, 5.524106634, 5.67722686, 5.712562297, 5.500549676, 5.582999029, 5.641891423, 5.441657281, 5.441657281),
  Lift_10 = c(4.48175693, 4.53476063, 4.53476063, 4.54064993, 4.48175693, 4.50531413, 4.48175693, 4.48764623, 4.434642538, 4.469978335, 4.440531837, 4.481756934, 4.493535533, 4.452310436, 4.452310436, 4.487646233, 4.452310436, 4.52887133, 4.511203432, 4.481756934, 4.381638842, 4.41108534, 4.369860243, 4.346303045, 4.358081644, 4.305077948, 4.316856547, 4.358081644, 4.240295654)
)

p1 <- ggplot(data, aes(x = AnzahlVariablen, y = AUROC)) +
  geom_smooth(method = "loess", colour = "#999999", se = FALSE) +
  labs(title = "AUROC vs. Number of Variables", x = "Number of Variables", y = "AUROC") +
  theme_minimal()

p2 <- ggplot(data, aes(x = AnzahlVariablen, y = Lift_1)) +
  geom_smooth(method = "loess", colour = "#E6F900", se = FALSE) +
  labs(title = "Lift_1 vs. Number of Variables", x = "Number of Variables", y = "Lift_1") +
  theme_minimal()

p3 <- ggplot(data, aes(x = AnzahlVariablen, y = Lift_5)) +
  geom_smooth(method = "loess", colour = "#999999", se = FALSE) +
  labs(title = "Lift_5 vs. Number of Variables", x = "Number of Variables", y = "Lift_5") +
  theme_minimal()

p4 <- ggplot(data, aes(x = AnzahlVariablen, y = Lift_10)) +
  geom_smooth(method = "loess", colour = "#E6F900", se = FALSE) +
  labs(title = "Lift_10 vs. Number of Variables", x = "Number of Variables", y = "Lift_10") +
  theme_minimal()

# Alle vier Plots in einem 2x2 Grid anzeigen
grid.arrange(p1, p2, p3, p4, ncol = 2)

## Verteilung Datentypen----
file_path <- "type.txt"  
data_types <- readLines(file_path)

data_types <- gsub("\\([^()]*\\)", "", data_types) 
data_types <- gsub("\\s+CHAR$", "", data_types)    

type_counts <- table(data_types)

type_df <- data.frame(Type = names(type_counts), Frequency = as.integer(type_counts))
type_df <- type_df[type_df$Frequency > 2, ]

type_df$Percentage <- (type_df$Frequency / sum(type_df$Frequency)) * 100
type_df <- type_df['Frequency' > 2]

ggplot(type_df, aes(x = Type, y = Percentage, fill = Type)) +
  geom_bar(stat = "identity", color = "black") +
  theme_minimal() +
  labs(title = "Verteilung der Datentypen in dem Datensatz",
       x = "Datentyp",
       y = "Prozentsatz") +
  theme(legend.position = "none", 
        axis.text.x = element_text(angle = 45, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 100)) +
  scale_fill_uchicago('dark')
