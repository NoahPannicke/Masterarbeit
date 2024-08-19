## function to return variables that correlate highly with others
get_high_correlation <- function(input_data, threshold = corr_threshold ){
  
  ## Features with no variance
  sd_vec      <- sapply(input_data, sd)
  feat_no_var <- names(sd_vec[sd_vec == 0])
  
  ## Find highly correlated (almost linearly dependent) features to remove 
  ## Note: Features with no variance are excluded bc. no correlation can be 
  ## computed as well as the TARGET to not bias the results of findCorrelation
  cormat         <- cor(input_data[,names(input_data)[names(input_data) %nin% c("TARGET", feat_no_var)],
                                   with = FALSE])
  feat_lin_dep   <- caret::findCorrelation(cormat, names = TRUE, cutoff = threshold, exact = TRUE, verbose = TRUE)
  
  return(feat_lin_dep)
  
}

load_prepared_data <- function(){
  
  if(rd_score == rd_test_iter){
    
    rd_comb_iter <- c(rd_train_iter, rd_test_iter)
  }else{
    
    rd_comb_iter <- c(rd_train_iter, rd_test_iter, rd_score)
  }
  
  # load data
  train_data_iter <- do.call(rbind, lapply(rd_comb_iter, function(x){
    
    print("load_snapshot_rd")
    
    ## Construct file name
    file_name <- tolower(paste0('snapshot_', snapshot_pattern, "_", x, "_reduced.csv"))
    
    print("PFAD ok")
    
    ## Load data
    tic()
    dataset   <- data.table::fread(file.path(snapshot_dir, file_name),
                                   stringsAsFactors = TRUE, nrows = dt_rows)
    toc()
    
    print("Daten laden ok")
    
    return(dataset)
    
  }))
  
  print("train_data_iter")
  
  ## Build subset of data (Alter, BL, Versart, Versdauer)
  if(is.null(train_subset_list)){
    
    x_subset <- train_data_iter
    
  }else{
    
    ## Create vart-idx first
    if(is.null(train_subset_list$vartklasse)){
      
      vart_idx <- train_data_iter$FVER_VERSSTATUS_BEZ %in% "Mitglieder"
      
    }else{
      
      vart_idx <- train_data_iter$FVER_VARTKLASSE_BEZ %in% train_subset_list$vartklasse
      
    }
    x_subset <- train_data_iter[  FVER_ALTER >= train_subset_list$age_min & 
                                    FVER_ALTER <= train_subset_list$age_max &   
                                    FVMZ_BEGINN_DIFF_TAGE >= train_subset_list$dauer_min &
                                    FVMZ_BEGINN_DIFF_TAGE <= train_subset_list$dauer_max &
                                    vart_idx &
                                    FVER_BUNDESLAND_CODE_3 %in% train_subset_list$bundesland]
    if('FMER_BONUSAPP_STICHTAG_FLAG' %in% names(x_subset)) x_subset <- x_subset[FMER_BONUSAPP_STICHTAG_FLAG == 0,]
    
  }
  
  # Redefine target name
  names(x_subset)[names(x_subset)==target] <- "TARGET"
  
  ## Remove irrelevant features
  if(!is.null(var_sel)){
    
    sub <- str_replace_all(var_sel[var_sel %nin% target]," ","")
    # Entfernen Variablen nicht in DB
    sub <- sub[sub %in% names(x_subset)]
    # Veriablen welche immer benötigt werden
    to_add <- c('TARGET','META_STICHTAG','DVER_PID')
    # Prüfung ob Variable im subset enthalten
    sub <- c(sub, to_add[to_add %nin% sub])
    
    x_subset <- x_subset[,..sub]
  }
  
  ## Alle Spaltennamen der Daten
  col_names     <- names(x_subset)
  
  ## Exclude variables that have no variance (siehe nvz-analyse)
  excl_zero_var <- c("FVER_PLZ_DEAKTIV_FLAG", "FVMZ_NICHT_KV_FLAG", "FVMZ_ALLE_FLAG", 
                     "FVMZ_UNBEKANNT_FLAG", "FVMZ_NICHT_KV_FLAG_STICHTAG", 
                     "FVMZ_ALLE_FLAG_STICHTAG", "FVMZ_UNBEKANNT_FLAG_STICHTAG", 
                     "FMER_KUNDENZ_STICHTAG_ANZ", 
                     "FVMZ_ARBG_BEZ_2", "FVMZ_ARBG_BEZ_GROUP_FLAG")
  
  
  # Exclude columns based on start and end pattern
  start_excl    <- #startsWith(col_names, "FVKU_") |
    #startsWith(col_names, "FVAQ_") |
    #startsWith(col_names, "MWM_")  |
    startsWith(col_names, "V_")    |
    #startsWith(col_names, "META_") |
    startsWith(col_names, "NIKA_") |
    startsWith(col_names, "DVER_STERB") 
  
  end_excl      <- endsWith(col_names, "_DAT") |
    endsWith(col_names, "_BEZ")  |
    #endsWith(col_names, "_CODE") |
    endsWith(col_names, "_ID")   |
    endsWith(col_names, "DATUM") |
    endsWith(col_names, "GEOMARKET")
  
  pattern_excl <- col_names[start_excl | end_excl | col_names %in% excl_zero_var]
  pattern_excl <- pattern_excl[pattern_excl %nin% target]
  
  ## Keep numerics and integers only
  data_classes  <- sapply(x_subset, class)
  num_keep      <- names(data_classes[(data_classes == "numeric" | data_classes == "integer")])
  
  ## Features to keep without target encoding
  ## Variablen welche immer ausgeschlossen werden sollen
  always_excl <- c("FVMZ_ENDE_DIFF_TAGE", "FVMZ_ENDE_DAT",
                   "FVMZ_FOLGEKK_BEZ", "FVMZ_FOLGEKK_DAT_MAX",
                   'FMER_DIGITAL_ANZ')
  
  cols_keep <- num_keep[num_keep %nin% unique(c(always_excl, pattern_excl))]
  cols_keep <- c(cols_keep,'META_STICHTAG','DVER_PID')
  
  ## For target encoding: add columns for target encoding
  if(do_te) cols_keep <- unique(c(cols_keep, factors_te))
  
  x_subset <- x_subset[, cols_keep, with = FALSE]
  
  target_enc_list <- "default" 
  high_corr_names <- "default"
  
  x_subset$META_STICHTAG <- as.Date(x_subset$META_STICHTAG)
  
  test_data <- x_subset[META_STICHTAG == as.Date(paste0(rd_test_iter,'-01')) + months(1) - days(1),]
  test_data[,META_STICHTAG := NULL]
  
  score_data    <- x_subset[META_STICHTAG == as.Date(paste0(rd_score_iter,'-01')) + months(1) - days(1),]
  score_data_st <- unique(score_data$META_STICHTAG)
  score_data[,META_STICHTAG := NULL]
  
  x_subset <- x_subset[META_STICHTAG != as.Date(paste0(rd_test_iter,'-01')) + months(1) - days(1),]
  x_subset[,c('DVER_PID','META_STICHTAG') := NULL]
  
  if(do_te){
    
    print("do_te")
    ## Get target encoded train date with mapping rules

    if(length(target_threshold) == 1) target_threshold <- rep(target_threshold[1], length(factors_te))
    
    
    ## Set Data Table (faster than data frame)
    setDT(x_subset)
    
    ## rules_list: used to provide values for target encoding in test set
    rules_list        <- vector(mode = "list", length = length(factors_te))
    names(rules_list) <- factors_te
    
    ## Regroup groups with few observations to group "Rest"
    for(k in 1:length(factors_te)){
      
      new_te_feature <- paste0(factors_te[k], "_TARGET_ENC")
      
      ## Create target encoded feature
      x_subset[,(new_te_feature) := mean(TARGET), by = x_subset[[factors_te[k]]]]
      
      ## Find Categories with less observations than 'min_size'
      ## Note: data table uses the name 'train_data_iter' in te_agg as grouping column
      ## based on the the command in the by-statement
      te_agg          <- x_subset[, .(.N, TE = mean(TARGET)), by = x_subset[[factors_te[k]]]][order(N, decreasing = TRUE)]
      rest_group      <- as.character(te_agg[N < target_threshold[k]]$x_subset)
      te_agg$Group    <- as.character(te_agg$x_subset)
      te_agg[x_subset %in% rest_group, Group := "RESTGROUP_TE"]
      
      
      ## Summarize categories if necessary 
      if(length(rest_group) > 0){
        
        
        ## Adjust grouping rules
        rest_group_te <- x_subset[x_subset[[factors_te[k]]] %in% rest_group, mean(TARGET)]
        te_agg[x_subset %in% rest_group, TE:= rest_group_te]
        
        ## Replace all values having a count below min_size with rest_group_te
        x_subset[x_subset[,get(factors_te[k])] %in% rest_group,c(new_te_feature) := rest_group_te]
        
      }  ## End of If-Loop
      
      names(te_agg)[names(te_agg) == "TE"]       <- new_te_feature
      names(te_agg)[names(te_agg) == "x_subset"] <- factors_te[k]
      
      rules_list[[k]] <- te_agg  
      
    }
    
    ## Anbindung des TE an die Testdaten 
    for(b in 1:length(rules_list)){
      test_data  <- merge(test_data,rules_list[[b]][,c(1,3)],by = colnames(rules_list[[b]])[1], all.x = T)
      score_data <- merge(score_data,rules_list[[b]][,c(1,3)],by = colnames(rules_list[[b]])[1], all.x = T)
      
      test_data[,colnames(rules_list[[b]])[1] := NULL] 
      score_data[,colnames(rules_list[[b]])[1] := NULL] 
    }
    
    target_enc_list <- list(train_data_te = x_subset[,-factors_te, with = FALSE], rules_list = rules_list)
    train_data_iter <- target_enc_list$train_data_te
    print("do_te ENDE")
    
  }else{
    train_data_iter <- x_subset}
  # End do_te
  
  # Exclude high correlated features
  if (do_correlation){
    
    ## Features with no variance
    sd_vec      <- sapply(train_data_iter, sd)
    feat_no_var <- names(sd_vec[sd_vec == 0 | is.na(sd_vec)])
    
    ## Find highly correlated (almost linearly dependent) features to remove 
    ## Note: Features with no variance are excluded bc. no correlation can be 
    ## computed as well as the TARGET to not bias the results of findCorrelation
    cormat         <- cor(train_data_iter[,names(train_data_iter)[names(train_data_iter) %nin% c("TARGET", feat_no_var)],
                                     with = FALSE])
    feat_lin_dep   <- caret::findCorrelation(cormat, names = TRUE, cutoff = corr_threshold, exact = TRUE, verbose = TRUE)

    train_data_iter <- train_data_iter[,names(train_data_iter) %nin% high_corr_names, with = FALSE]	
    
  }
  
  return(list(train_data_iter=train_data_iter,
              test_data_iter =test_data,
              score_data_iter=score_data,
              target_enc_list=target_enc_list,
              high_corr_names=high_corr_names,
              score_data_st  =score_data_st,
              db_col_name    =db_column))
  
  rm(train_data_iter, test_data, test_data_iter, score_data)
}

calc_eval_metrics <- function(pred_test,y_test,eval_crit_list){
  
  auroc <- tryCatch(pROC::auc(predictor = pred_test, 
                              response = y_test,
                              quiet    = TRUE),
                    error = function(e){
                      -1
                    })
  
  
  ### Construct evaluation data frame
  eval_df        <- data.frame(AUROC = as.numeric(auroc))
  
  ## Lift test set
  lift_test         <- lapply(eval_crit_list$Lift, 
                              function(x){
                                if(is.factor(y_test)) labels <- as.integer(as.character(y_test))
                                
                                lift <- cbind(pred_test, y_test)
                                lift <- lift[order(-lift[, 1]), ]
                                lift <- as.numeric(mean(lift[1:round(nrow(lift)*x, 0), 2])/mean(lift[,2]))
                                round(lift, digits = 3)
                                ifelse(is.nan(lift), -1, lift) 
                              })
  ## Lift FN Test
  #sum(y_test)
  lift_rest         <- lapply(eval_crit_list$Lift, 
                              function(x){
                                if(is.factor(y_test)) labels <- as.integer(as.character(y_test))
                                
                                lift <- cbind(pred_test, y_test)
                                lift <- lift[order(-lift[, 1]), ]
                                lift <- paste0(sum(lift[1:round(nrow(lift)*x, 0), 2]),' von ',sum(y_test),' - ', round(sum(lift[1:round(nrow(lift)*x, 0), 2])/sum(y_test),3)*100,'%')
                              })
  
  
  names(lift_test) <- paste0("Lift_", eval_crit_list$Lift*100)
  
  ## Combine all evaluation criteria  
  eval_df <- cbind(eval_df, lift_test, lift_rest)
  
  return(eval_df)
}