#' Skritp zur Analyse der Features zur Erstellung
#' des Produktaffinitätsmodells 
rm(list = ls())
# Laden der Funktionen 
source("Code/functions_ma.R")

options(warn=-1, error=NULL)
memory.limit(size=1000000)


# Laden der Pakete
library(data.table)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(ggpubr)
library(stringr)
library(RColorBrewer)
library(ggsci)
library(grid)
library(corrplot)

# ----Auswahl relevanter Features für Analyse----
rel_fea <- c(# Stammdaten
              'FVER_ALTER',
              'FVMZ_BEGINN_DIFF_TAGE',
              'FVER_GESCHLECHT',
              'FVER_BUNDESLAND_CODE_3',
              'DVER_NATION_BEZ',
              'FVER_VARTGRUPPE_BEZ',
              'FMER_MEINEAOK_ANZ',
              'DTAE_TAETIGKEIT_BEZ',
              'FMER_DIGITAL_ANZ',
              'DARB_ORT',
              'DARB_BRANCHE_CODE',
              'FLSP_GESKONTO_ANZ',
              'FLSP_ZUZAHLUNG_REL',
              'FETZ_ENTGELT_JAHR1_TAGE_SUM',
              'FFAL_LETZTE_KH_TAGE',
              'FLSP_ZAHLBETRAG_SUM',
              'FLSP_LPO_ANZ',
              'FLSP_ZAHLBETRAG_PRO_LPO',
              'FAPO_ARZNEI_BETRAG_SUM',             
              
              # Configdaten
              'META_STICHTAG',
              'FMER_BONUSAPP_STICHTAG_FLAG',
              'V_FLSP_AOK_BONUS_AUSZ_FLAG_6M')

rel_fea <- c(# Stammdaten

  'FLSP_LPO_ANZ',
  'FLSP_ZAHLBETRAG_PRO_LPO',

  # Configdaten
  'META_STICHTAG',
  'FMER_BONUSAPP_STICHTAG_FLAG',
  'V_FLSP_AOK_BONUS_AUSZ_FLAG_6M')


# ----Parameter----
stichtag <- "('31.12.22')"
use_branche <- F

# ----Laden der Grundtabellen----
se_features     <- fread("daten/31.12.22.csv", encoding = 'Latin-1')
se_features     <- se_features[,..rel_fea]

branche_code    <- fread("daten/branche_code.csv",sep = ';',encoding = 'Latin-1')

# add leading zero to branche_code
branche_code$BRANCHE_CODE <- str_pad(branche_code$BRANCHE_CODE, width = 4, side = "left", pad = "0")

ag_id           <- fread("daten/arbeitgeber_id.csv",sep = ';',encoding = 'Latin-1')

# ----NA Handling----
na_agg <- colSums(is.na(se_features))
na_agg

# ----Outlierdetection----
outlier_detection <- function(data){
  
  # Outlier Detection
  outlier <- data[,lapply(.SD, function(x) quantile(x, c(0.25, 0.75), na.rm = T)), .SDcols = names(data)]
  
  # Interquartile Range
  outlier[,IQR := V75 - V25]
  
  # Outlier Detection
  outlier[,V25 := V25 - 1.5*IQR]
  outlier[,V75 := V75 + 1.5*IQR]
  
  # Outlier Detection
  outlier <- outlier[,lapply(.SD, function(x) sum(x < V25 | x > V75, na.rm = T)), .SDcols = names(data)]
  
  return(outlier)
}

outlier_cleaned_data <- function(data, ignore_columns = NULL) {
  # Kopie des Datenrahmens erstellen, um Originaldaten unverändert zu lassen
  cleaned_data <- data
  
  # Bestimme die Spalten, die für die Ausreißerbereinigung berücksichtigt werden sollen
  relevant_columns <- setdiff(names(data), ignore_columns)
  
  # Durchlauf durch die relevanten Spalten im Datenrahmen
  for (col_name in relevant_columns) {
    # Berechnung der Quartile
    quartiles <- quantile(data[[col_name]], probs = c(0.25, 0.75), na.rm = TRUE)
    Q1 <- quartiles[1]
    Q3 <- quartiles[2]
    
    # Berechnung des Interquartilabstands
    IQR <- Q3 - Q1
    
    # Festlegung der Ausreißergrenzen
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    
    # Entfernen der Ausreißer unter Berücksichtigung der relevanten Spalten
    cleaned_data <- cleaned_data[cleaned_data[[col_name]] >= lower_bound & cleaned_data[[col_name]] <= upper_bound, ]
  }
  
  # Rückgabe des bereinigten Datenrahmens
  return(cleaned_data)
}

## Ordner für Ergebnissoutput
only_num <- !unlist(lapply(se_features, is.numeric))

cols_non_num <- names(se_features[,..only_num])

cols_non_num <- c(cols_non_num,
                  'CRM_ZUSATZ_ANZ',
                  'FEPD_EPOST_12M_ANZ',
                  'FLSP_GESKONTO_ANZ',
                  'FMER_DIGITAL_ANZ',
                  'FMER_MEINEAOK_ANZ',
                  'V_FLSP_AOK_BONUS_AUSZ_FLAG_6M')

always_exc <- c('META_STICHTAG',
                'DARB_BRANCHE_CODE',
                'DARB_ARBEITGEBER_ID',
                'FMER_BONUSAPP_STICHTAG_FLAG',
                'BRANCHE_BEZ',
                'BRANCHE_CODE_BUCHSTABE',
                'ARBEITGEBER',
                'NAME_2',
                'DTAE_TAETIGKEIT_BEZ')

out_det <- c('FLSP_ZAHLBETRAG_PRO_LPO',
             'FLSP_LPO_ANZ')

cols_non_num <- cols_non_num[cols_non_num %in% colnames(data)]
always_exc   <- always_exc[always_exc %in% colnames(data)]

# Erstellen Unterordner zum Speichern der Ergebniss
dateDIR   <- paste0('analyseergebnisse_',as.character(format(Sys.time(), "%Y_%m_%d_%H_%M")))
outputDIR <- file.path(res_path, dateDIR)
if (!dir.exists(outputDIR)) {dir.create(outputDIR)}

## ----Branchen- und Arbeitgeberanalyse----
if(use_branche == T){
  
  # extraction of non-unique Branchen
  non_uni_bra        <- branche_code[,.(N = .N), BRANCHE_CODE][order(N, decreasing = T)][N > 1, BRANCHE_CODE]
  branche_code_edit  <- branche_code[BRANCHE_CODE %in% non_uni_bra & (is.na(BRANCHE_CODE_BUCHSTABE) | BRANCHE_CODE_BUCHSTABE != 'n/v') & BRANCHE_CODE_BUCHSTABE != '',]
  branche_code       <- branche_code[!(BRANCHE_CODE %in% non_uni_bra),]
  branche_code       <- rbind(branche_code,branche_code_edit)
  
  data <- unique(merge.data.table(data, branche_code,
                              by.x = c('DARB_BRANCHE_CODE'),
                              by.y = c('BRANCHE_CODE'),
                              all.x = T))
  
  
  data <- unique(merge.data.table(data, ag_id,
                              by.x = c('DARB_ARBEITGEBER_ID'),
                              by.y = c('AG_ID'),
                              all.x = T))
  
  # Arbeitgeber
  flags <- c('0','1')
  titles <- c('Nicht-User','User')
  
  data[is.na(NAME_1),NAME_1 := 'n/v']
  
  setnames(data,c('NAME_1'),c('ARBEITGEBER'))
  
  erg_plot <- vector(mode = 'list',length = length(flags))
  label_x <- c('Arbeitgeber Nicht-Nutzer BonusApp','Arbeitgeber Nutzer BonusApp')
  for(i in 1:length(flags)){
    
    prio_ag <- data[META_STICHTAG == max(data$META_STICHTAG) & get(flag) == flags[[i]] & ARBEITGEBER != 'n/v',.(Anzahl = .N),by='ARBEITGEBER'][order(Anzahl,decreasing = T)][1:10]
    
    prio_ag <- prio_ag[,Anteil := round((Anzahl/sum(prio_ag$Anzahl))*100,2)]
    
    ag_plot <- ggplot(prio_ag,aes(x=reorder(ARBEITGEBER,Anteil),y=Anteil,fill=factor(ARBEITGEBER))) +
      geom_bar(stat = 'identity', fill = '#999999') +
      xlab(paste(titles[[i]],flag)) +
      geom_text(aes(label = Anteil),
                vjust = 0.5,hjust = -0.1, position = "identity", color ="black") +
      guides(fill='none') +
      coord_flip() +
      scale_y_continuous(limits = c(0,round(max(prio_ag$Anteil))+2.5)) +
      theme_bw() +
      labs(y = 'Anteil in %',x = label_x[[i]])
    
    erg_plot[[i]] <- ag_plot
  }
  
  full_AG <- grid.arrange(erg_plot[[1]],
               erg_plot[[2]],
               ncol=2,
               top = textGrob("Top Arbeitgeber nach Nutzerstatus",
                              gp = gpar(fontsize = 16, fontface = "bold"), x = 0, hjust = 0))
  
  ggsave(paste0('Top_Arbeitgeber.png'),
         plot = full_AG,
         path = outputDIR,
         width = 35,
         height = 15,
         units = 'cm',
         dpi = 650)
  
  # Branchen
  data[is.na(DARB_BRANCHE_CODE),DARB_BRANCHE_CODE := 'n/v']
  
  data_branche_no_nv <- data[DARB_BRANCHE_CODE != 'n/v' & DARB_BRANCHE_CODE != 0,]
  
  label_x <- c('Branche Nicht-Nutzer BonusApp','Branche Nutzer BonusApp')
  
  br_plot <- list()
  for(i in 1:length(flags)){
    prio_branchen <- data_branche_no_nv[META_STICHTAG == max(data_branche_no_nv$META_STICHTAG) & get(flag) == flags[[i]] & DARB_BRANCHE_CODE != 'n/v',.(Anzahl = .N),by='BRANCHE_BEZ'][order(Anzahl,decreasing = T)][1:10]
    
    prio_branchen[,Anteil := round((Anzahl/sum(prio_branchen$Anzahl))*100,2)]
    
    br_plot <- ggplot(prio_branchen,aes(x=reorder(BRANCHE_BEZ,Anteil),y=Anteil,fill=factor(BRANCHE_BEZ))) +
      geom_bar(stat = 'identity', fill = '#999999') +
      xlab(paste(titles[[i]],flag)) +
      geom_text(aes(label = Anteil),
                vjust = 0.5,hjust = -0.1, position = "identity", color ="black") +
      guides(fill='none') +
      coord_flip() +
      scale_y_continuous(limits = c(0,round(max(prio_branchen$Anteil))+2.5)) +
      theme_bw() +
      labs(y = 'Anteil in %',x = label_x[[i]])
    
    erg_plot[[i]] <- br_plot
  }
  
  full_ABR <- grid.arrange(erg_plot[[1]],
                           erg_plot[[2]],
                          ncol=2,
                          top = textGrob("Top Branchen nach Nutzerstatus",
                                         gp = gpar(fontsize = 16, fontface = "bold"), x = 0, hjust = 0))
  
  ggsave(paste0('Branchen.png'),
         plot = full_ABR,
         path = outputDIR,
         width = 35,
         height = 15,
         units = 'cm',
         dpi = 650)
  
  # Tätigkeiten
  data[is.na(DTAE_TAETIGKEIT_BEZ),DTAE_TAETIGKEIT_BEZ := 'n/v']
  
  data_taetigkeit_no_nv <- data[DTAE_TAETIGKEIT_BEZ != 'n/v' & DTAE_TAETIGKEIT_BEZ != 0 & DTAE_TAETIGKEIT_BEZ != '',]
  
  label_x <- c('Tätigkeit Nicht-Nutzer BonusApp','Tätigkeit Nutzer BonusApp')
  
  tae_plot <- list()
  for(i in 1:length(flags)){
    prio_tae <- data_taetigkeit_no_nv[META_STICHTAG == max(data_taetigkeit_no_nv$META_STICHTAG) & get(flag) == flags[[i]] & DTAE_TAETIGKEIT_BEZ != 'n/v',.(Anzahl = .N),by='DTAE_TAETIGKEIT_BEZ'][order(Anzahl,decreasing = T)][1:10]
    
    prio_tae[,Anteil := round((Anzahl/sum(prio_tae$Anzahl))*100,2)]
    
    br_plot <- ggplot(prio_tae,aes(x=reorder(DTAE_TAETIGKEIT_BEZ,Anteil),y=Anteil,fill=factor(DTAE_TAETIGKEIT_BEZ))) +
      geom_bar(stat = 'identity', fill = '#999999') +
      xlab(paste(titles[[i]],flag)) +
      geom_text(aes(label = Anteil),
                vjust = 0.5,hjust = -0.1, position = "identity", color ="black") +
      guides(fill='none') +
      coord_flip() +
      scale_y_continuous(limits = c(0,round(max(prio_tae$Anteil))+2.5)) +
      theme_bw() +
      labs(y = 'Anteil in %',x = label_x[[i]])
    
    erg_plot[[i]] <- br_plot
  }
  
  full_ABR <- grid.arrange(erg_plot[[1]],
                           erg_plot[[2]],
                           ncol=2,
                           top = textGrob("Top Taetigkeiten nach Nutzerstatus",
                                          gp = gpar(fontsize = 16, fontface = "bold"), x = 0, hjust = 0))
  
  ggsave(paste0('Taetigkeiten.png'),
         plot = full_ABR,
         path = outputDIR,
         width = 35,
         height = 15,
         units = 'cm',
         dpi = 650)
}

# Make list of variable names to loop over.
var_list    = rel_fea[rel_fea %nin% c(cols_non_num,always_exc,'META_STICHTAG')]
var_list_sp = rel_fea[rel_fea %nin% c(always_exc,'META_STICHTAG')]

# Durchläuft alle Metriken
# - mean
# - median
for(k in 1:length(metric)){
  
  # Berechnung der Metrik 
  # gruppiert nach Produktnutzer
  num_data <- data[, lapply(.SD, as.numeric), .SDcols=names(data[,
                                                     .SD,
                                                     .SDcols=!c(cols_non_num,always_exc)])]
  
  num_data <- cbind(num_data,META_STICHTAG = as.character(data[,META_STICHTAG]))
  
  # mean_population <-  num_data[,lapply(.SD,base::get(metric[[k]]),na.rm=T),
  #                                 by = c(x_input),
  #                                 .SDcols=names(num_data[,
  #                                                           .SD,
  #                                                           .SDcols=!c(cols_non_num,always_exc)])][order(get(x_input))]

  mean_population <-        num_data[,lapply(.SD,base::get(metric[[k]]),na.rm=T),
                                  by = c(x_input)][order(get(x_input))]
  mean_population$META_STICHTAG <- as.POSIXct(mean_population$META_STICHTAG)
  
  #mean_population <- mean_population[,1:(length(mean_population)-1)]
  
  mean_population[,GROUP:='Population']
  
  # Sample
  num_data <- data[get(flag) == '1', lapply(.SD, as.numeric), .SDcols=names(data[,
                                                                 .SD,
                                                                 .SDcols=!c(cols_non_num,always_exc)])]
  
  num_data <- cbind(num_data,META_STICHTAG = as.character(data[,META_STICHTAG]))
  
  mean_sample     <-        num_data[,lapply(.SD,base::get(metric[[k]]),na.rm=T),
                                     by = c(x_input)][order(get(x_input))]
  mean_sample$META_STICHTAG <- as.POSIXct(mean_sample$META_STICHTAG)
  
  #mean_sample <- mean_sample[,1:(length(mean_sample)-1)]
  
  mean_sample[,GROUP:='Sample']
  
  # Combine
  mean_comb <- rbind(mean_population,mean_sample)
  
  # Make lineplots
  plot_list = list()
  for (i in 1:length(var_list)) {
      
     p = ggplot(mean_comb, aes_string(x=x_input, y=var_list[[i]][1], color = 'GROUP')) +
         geom_point(size=3) + geom_line() +
       theme_bw()
    
    plot_list[[i]] = p
    
  }
  # Speichern des Plots
  comb_plots <- do.call(grid.arrange, plot_list)
  ggsave(paste0(metric[[k]],'_comparison.jpg'),
         plot = comb_plots,
         path = outputDIR,
         width = 70,
         height = 35,
         units = 'cm',
         dpi = 650)
  }
  

  
  # Einschränkung auf 1 ST
  data <- data[META_STICHTAG == max(data$META_STICHTAG),]
  
  ## ----Korrelationsanalyse----
  # Korrelationsmatrix
  cor_data <- cor(data[,..var_list], use = 'pairwise.complete.obs')
  
  cor.mtest <- function(mat) {
    mat <- as.matrix(mat)
    n   <- ncol(mat)
    p.mat<- matrix(NA, n, n)
    diag(p.mat) <- 0
  
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        
        tmp <- cor.test(mat[, i], mat[, j])
        p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
        
      }
    }
    colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
    p.mat
  }
  
  # matrix of the p-value of the correlation
  p.mat <- cor.mtest(data[,..var_list])
  
  col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
  
  png(paste0(outputDIR,'/Korrelationsanalyse.png'), width = 900, height = 900)
  cor_analyse <- corrplot(cor_data, method="color", col=col(200),  
                           type="upper", order="hclust", 
                           title = "Korrelationsanalyse mit Signifikanzniveau 0.01",
                           addCoef.col = "black", # Add coefficient of correlation
                           tl.col="black", tl.srt=45, #Text label color and rotation
                           # Combine with significance
                           p.mat = p.mat, sig.level = 0.01,
                           number.cex = 0.9, # Text size
                           # hide correlation coefficient on the principal diagonal
                           diag=FALSE,
                           mar = c(0, 0, 1, 0)
                  )
  dev.off()
  
  ## Outlier 
  #data <- outlier_cleaned_data(data = data, ignore_columns = c(cols_non_num,always_exc,'GROUP'))
  
  ## ----Histogram, Violine Plot----
  data[get(flag) == '0',GROUP:='Population']
  data[get(flag) == '1',GROUP:='Sample']
  
  colnames(data) <- make.unique(names(data))
  
  # Ordern zum Speichern der Plots
  dateDIR        <- paste0('Histogram')
  outputDIR_hist <- file.path(outputDIR, dateDIR)
  if (!dir.exists(outputDIR_hist)) {dir.create(outputDIR_hist)}
  
  dateDIR        <- paste0('Violine')
  outputDIR_vio <- file.path(outputDIR, dateDIR)
  if (!dir.exists(outputDIR_vio)) {dir.create(outputDIR_vio)}
  
  dateDIR        <- paste0('Box')
  outputDIR_box <- file.path(outputDIR, dateDIR)
  if (!dir.exists(outputDIR_box)) {dir.create(outputDIR_box)}
  
  # Ordner zum Speichern der kategorialen Variablen
  dateDIR        <- paste0('Categorial')
  outputDIR_cat  <- file.path(outputDIR, dateDIR)
  if (!dir.exists(outputDIR_cat)) {dir.create(outputDIR_cat)}
  
  # Make histogram
  #plot_list = list()
  for (i in 1:length(var_list_sp)) {
    
    # ----Kategoriale Merkmale----
    if(var_list_sp[[i]] %in% cols_non_num){
      
      print(var_list_sp[[i]])
      
      col <- var_list_sp[[i]]
      
      data[get(beo_flag) == '1', PLOT_FLAG := 'Nutzer']
      data[get(beo_flag) != '1', PLOT_FLAG := 'Nicht-Nutzer']
      
      data <- data[FVER_GESCHLECHT != 'D',]
      
      data$PLOT_FLAG <- as.factor(data$PLOT_FLAG)
  
      # Top 5 Kategoriale Ausprägungen
      top_5 <- data[PLOT_FLAG == 'Nutzer',.(N=.N),by=get(var_list_sp[[i]])][order(N,decreasing = T)][1:5]
      top_5 <- as.character(top_5$get)
      
      temp <- data[get(var_list_sp[[i]]) %in% top_5,]
      
      agg_dt <- temp[,.(count = .N), by = c(var_list_sp[[i]], 'PLOT_FLAG')][,prop := paste0(round(count/sum(count),3)*100,'%'), by = c('PLOT_FLAG')]
      
      total <- 
        ggplot(agg_dt, aes_string(var_list_sp[[i]], 'count', fill = 'PLOT_FLAG')) + 
        geom_bar(stat="identity",position = 'dodge') + 
        #scale_y_continuous(labels=scales::percent) +
        ggtitle(var_list_sp[[i]]) +
        scale_y_continuous(name="Anzahl", labels = scales::comma) +
        theme(axis.text.x = element_text(angle = 10),
              legend.title = element_blank()) +
        #stat_count(aes(label = paste(round(prop.table(..count..),4) * 100, "%", sep = "")),
        #           vjust = 1, geom = "text", position = "identity", color ="black") +
        geom_text(aes_string(var_list_sp[[i]], 'count', label = 'prop'), 
                  position = position_dodge(width = 0.95), size=4, vjust = -0.25) +
        scale_color_manual(values=c("#999999", "#E69F00"))+
        scale_fill_manual(values=c("#999999", "#E69F00"))
  
      ggsave(paste0('categorial_',var_list_sp[[i]],'.jpg'),
             plot = total,
             path = outputDIR_cat,
             width = 20,
             units = 'cm')
      
    }else{
    
    print(var_list_sp[[i]])
    k <- 1
    
    data[,grp.mean := lapply(.SD,base::get(metric[[k]]),na.rm=T),by = GROUP,.SDcols=var_list_sp[[i]][1]]
    mu <- unique(data[,.(GROUP,grp.mean)])
    data[,`:=`(grp.mean=NULL)]
    
    # notwendig für Violine Plot
    median_var     <- aggregate(get(var_list_sp[[i]]) ~  GROUP, data, median, na.rm=T, na.action=NULL)
    median_var[,2] <- ifelse(median_var[,2] > 10,round(median_var[,2],0),round(median_var[,2],5))
    setnames(median_var,'get(var_list_sp[[i]])',var_list_sp[[i]])
    
    # ----Histogramm----
    # Freedman-Diaconis rule
    bw <- 2 * IQR(data[,get(var_list_sp[[i]])],na.rm = T) / length(data[!is.na(get(var_list_sp[[i]])),get(var_list_sp[[i]])])^(1/3)
    
    p = ggplot(data, aes_string(x=var_list_sp[[i]][1],color='GROUP',fill='GROUP')) +
      geom_histogram(aes(y=..density..),alpha=0.5,binwidth = bw) +
      geom_density(alpha=0.6) +
      facet_grid(GROUP ~ .)
  
    p + geom_vline(mu,mapping=aes(xintercept=grp.mean, color=GROUP),linetype='dashed') +
      scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
      scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
      theme_bw() +
      theme(legend.title=element_blank())
    
    ggsave(paste0('histogram_',var_list_sp[[i]],'.jpg'),
           path = outputDIR_hist,
           width = 20,
           units = 'cm')
    
  # ----Violine----
  if(var_list_sp[[i]] %in% out_det){
    data_out <- data[,.(get(var_list_sp[[i]]),GROUP)]
    setnames(data_out,'V1',var_list_sp[[i]])  
    data_out <- outlier_cleaned_data(data = data_out, ignore_columns = c('GROUP'))
    #data_out <- data_out[get(var_list_sp[[i]]) > 0,]
  }else{
    data_out <- data
  }
 
    
   ggplot(data_out,aes(x=GROUP,y=get(var_list_sp[[i]]), color=GROUP)) +
      geom_violin() +
      geom_boxplot(width=0.1) +
      ylab(var_list_sp[[i]]) +
      #scale_y_log10(labels=scales::comma) +
      scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
      scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
      #scale_y_continuous(trans='log10') +
      #coord_trans(x = 'log10') +
      labs(color=NULL) +
      #geom_text(aes(label = paste("Md: ",median_var[1,2]), y = min(median_var[,2]), x = 4),check_overlap = T) +
     annotate("text", y = min(median_var[,2]), x = 2.3, label = paste("Md: ",median_var[2,2]), color = 'orange') +
     annotate("text", y = max(median_var[,2]), x = 0.6, label = paste("Md: ",median_var[1,2]), color = 'darkgrey')#quantile(data[,get(var_list_sp[[i]][1])], probs = c(0.125,0.5,0.62,0.90))[['90%']]))
    
    ggsave(paste0('violine_',var_list_sp[[i]],'.jpg'),
           path = outputDIR_vio,
           width = 20,
           units = 'cm')
    
    # ----Boxplot----
    ggplot(data_out,aes(x=GROUP,y=get(var_list_sp[[i]]), color=GROUP)) +
      geom_boxplot() +
      ylab(var_list_sp[[i]]) +
      #scale_y_log10(labels=scales::comma) +
      scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9"))+
      scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
      labs(color=NULL) +
      #scale_y_continuous(trans='log10') +
      #coord_trans(x = 'log10') +
      #geom_text(aes(label = paste("Md: ",median_var[1,2]), y = min(median_var[,2]), x = 4),check_overlap = T) +
      annotate("text", y = (max(median_var[,2])-max(median_var[,2])*0.1), x = 2, label = paste("Md: ",median_var[2,2]), color = 'orange') +
      annotate("text", y = (max(median_var[,2])-max(median_var[,2])*0.1), x = 1, label = paste("Md: ",median_var[1,2]), color = 'darkgrey')
    
    ggsave(paste0('box_',var_list_sp[[i]],'.jpg'),
           path = outputDIR_box,
           width = 20,
           units = 'cm')
  
    }
  
}

# LOESS-Grafiken----
loess_func <- 'NIKA_AVG ~ FVER_ALTER'

filename <- 'alter_nika_ba.jpeg'

group_one <- "Männer"
group_two <- "Frauen"

stichtag <- c('2023-06-30 02:00:00')

user     <- se_features[get(V_FLSP_AOK_BONUS_AUSZ_FLAG_6M) == 1 &
                          META_STICHTAG == stichtag,]

user_one <- se_features[get(V_FLSP_AOK_BONUS_AUSZ_FLAG_6M) == 1 & FVER_GESCHLECHT == 'M' &
                          META_STICHTAG == stichtag,]

user_two <- se_features[get(V_FLSP_AOK_BONUS_AUSZ_FLAG_6M) == 1 & FVER_GESCHLECHT == 'W' &
                          META_STICHTAG == stichtag,]
# Grundgesamtheit
non_user         <- se_features[META_STICHTAG == stichtag]
non_user_sub     <- non_user[sample(nrow(non_user),300000)]

non_user_sub_one <- non_user_sub[FVER_GESCHLECHT == 'M',]

non_user_sub_two <- non_user_sub[FVER_GESCHLECHT == 'W',]


# Plot
loess.user           <- loess(loess_func, data = user) 
loess.user.one       <- loess(loess_func, data = user_one) 
loess.user.two       <- loess(loess_func, data = user_two) 
loess.non_user       <- loess(loess_func, data = non_user_sub)
loess.non_user.one   <- loess(loess_func, data = non_user_sub_one)
loess.non_user.two   <- loess(loess_func, data = non_user_sub_two)

kosten.user <- data.frame(ALTER            = 15:70,
                          BA_USER        = predict(loess.user,         newdata = 15:70),
                          BA_USER_ONE    = predict(loess.user.one,     newdata = 15:70),
                          BA_USER_TWO    = predict(loess.user.two,     newdata = 15:70),
                          ALL              = predict(loess.non_user,     newdata = 15:70),
                          ALL_ONE          = predict(loess.non_user.one, newdata = 15:70),
                          ALL_TWO          = predict(loess.non_user.two, newdata = 15:70))

kosten.user <- apply(kosten.user, 2, round)
kosten.user <- as.data.frame(kosten.user)

kosten.user <- na.omit(kosten.user)

setDT(kosten.user)
kosten.user <- kosten.user[,.(ALTER,BA_USER_ONE,BA_USER_TWO,ALL_ONE,ALL_TWO)]
kosten.user <- setnames(kosten.user, old = c('BA_USER_ONE','BA_USER_TWO','ALL_ONE','ALL_TWO'),
                        new = c(paste0('Nutzer_',group_one),paste0('Nutzer_',group_two),group_one,group_two))


kosten.user.melt <- melt(kosten.user, id.vars = 'ALTER')

ggplot(kosten.user, aes(ALTER)) +
  geom_line(aes(y = get(paste0('Nutzer_',group_one)), colour = paste0('Nutzer ',group_one)), size = 1) +
  geom_line(aes(y = get(paste0('Nutzer_',group_two)), colour = paste0('Nutzer ',group_two)), size = 1) +
  geom_line(aes(y = get(group_one), colour = group_one), size = 0.6) +
  geom_line(aes(y = get(group_two), colour = group_two), size = 0.6) +
  labs(x = "Alter",
       y = "Durchschnittlicher NIKA in Euro",
       color = "Legende") +
  ggtitle("Norm-Ist-Kosten-Analyse (NIKA) nach Alter und Geschlecht") +
  theme(legend.title=element_blank()) +
  scale_color_manual(name = 'Legende',
                     values=c('#FF6A6A',"#00BFFF","#F20034","#36648B"))

ggsave(filename,
       path = outputDIR,
       width = 25,
       height = 13,
       units = 'cm')

# Registrierungen Bonus-App----
data <- data.frame(
  Value = c(15399, 18156, 20882, 24944, 27030, 28942, 30521, 32064, 33672, 34865,
            36193, 37526, 38938, 40390, 42116, 44918, 46796, 48313, 49184, 51237,
            52612, 53933, 55100, 56720, 58321, 61680, 63148, 66254, 69853, 73147,
            74815, 76661, 78389, 80040, 81642, 83238, 84659, 86782, 89535, 93570,
            95778, 95840),
  Date = as.Date(c("2020-10-31", "2020-11-30", "2020-12-31", "2021-01-31", "2021-02-28",
                   "2021-03-31", "2021-04-30", "2021-05-31", "2021-06-30", "2021-07-31",
                   "2021-08-31", "2021-09-30", "2021-10-31", "2021-11-30", "2021-12-31",
                   "2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30", "2022-05-31",
                   "2022-06-30", "2022-07-31", "2022-08-31", "2022-09-30", "2022-10-31",
                   "2022-11-30", "2022-12-31", "2023-01-31", "2023-02-28", "2023-03-31",
                   "2023-04-30", "2023-05-31", "2023-06-30", "2023-07-31", "2023-08-31",
                   "2023-09-30", "2023-10-31", "2023-11-30", "2023-12-31", "2024-01-31",
                   "2024-02-29", "2024-03-31"), format="%Y-%m-%d")
)
setDT(data)

# Berechnung der monatlichen Neukunden
data[, NewCustomers := c(NA, diff(Value))]

# Scatter- und Linienplot
plot1 <- ggplot(data, aes(x = Date, y = Value)) +
  geom_point(color = '#E69F00', size = 3, alpha = 0.6) +
  geom_line(color = '#999999', size = 1) +
  labs(
    title = 'Nutzerentwicklung der Bonus-App',
    y = 'Nutzeranzahl',
    x = 'Jahr'
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  ) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y")

data[, Quarter := floor_date(Date, "quarter")]

# Aggregieren der Daten auf Quartalsbasis
data_aggregated <- data[, .(Value = last(Value)), by = Quarter]

# Berechnung der Neukunden für jedes Quartal
data_aggregated[, NewCustomers := c(NA, diff(Value))]

# Erstellen des Plots
plot2 <- ggplot(data_aggregated, aes(x = Quarter, y = NewCustomers)) +
  geom_bar(stat = 'identity', fill = "#999999", color = "black") +
  labs(
    title = 'Quartalsweise Neukunden der Bonus-App',
    y = 'Anzahl neuer Nutzer',
    x = 'Quartal'
  ) +
  theme_minimal(base_size = 14) +
  scale_x_date(date_labels = "%Y ") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability
# Kombinieren der Plots
grid.arrange(plot1, plot2, ncol = 1)

# Produktkorrelation----
prod <- fread("Deskriptive Analyse/prodcorr.csv")

prod <- prod[FVER_ALTER >= 15 & FVER_ALTER <= 75 & FVER_VERSSTATUS_BEZ == 'Mitglieder' & FVER_BUNDESLAND_CODE_3 %in% c('BLN','BRA','MVP')]

# BA-MAOK
prod_ba_ma <- prod[,.(BA_TN,B_MEINE_AOK_TN_FLAG)]

table(prod_ba_ma$BA_TN,prod_ba_ma$B_MEINE_AOK_TN_FLAG)
chisq.test(prod_ba_ma$BA_TN,prod_ba_ma$B_MEINE_AOK_TN_FLAG,correct = F)

# BA-BW
prod_ba_bw <- prod[,.(BA_TN,BW_TN)]

table(prod_ba_bw$BA_TN,prod_ba_bw$BW_TN)
chisq.test(prod_ba_bw$BA_TN,prod_ba_bw$BW_TN,correct = F)

# BA-GK
prod_ba_gk <- prod[,.(BA_TN,GK_TN)]

table(prod_ba_gk$BA_TN,prod_ba_gk$GK_TN)
chisq.test(prod_ba_gk$BA_TN,prod_ba_gk$GK_TN,correct = F)

library(rcompanion)
cohenW(prod_ba_ma$BA_TN,prod_ba_ma$B_MEINE_AOK_TN_FLAG)
cohenW(prod_ba_bw$BA_TN,prod_ba_bw$BW_TN)
cohenW(prod_ba_gk$BA_TN,prod_ba_gk$GK_TN)

library(sjPlot)
library(sjmisc)


sjPlot::plot_xtab(prod$BA_TN, prod$B_MEINE_AOK_TN_FLAG, margin = "row", bar.pos = "stack", coord.flip = TRUE)
# Kreuztabelle mit sjPlot darstellen
sjt.xtab(prod$BA_TN, prod$GK_TN,
         title = "Kreuztabelle: Nutzung der Bonus-App und des Gesundheitskontos",
         
         show.summary = TRUE,
         #show.n = TRUE,
         show.row.prc = TRUE,
         #show.col.prc = TRUE,
         var.labels = c("Bonus-App", "Meine AOK-App"),
         digits = 0,
         statistics = "cramer",
         file = "Vorstellung/ba_gk.html")

