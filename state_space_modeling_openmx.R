#--------------------------------------------------
#This code is for structural equation modeling of driver's state and behaviors in OpenMX and MRSS
#--------------------------------------------------------------------------------------------------

#packages
#-----------------------------------------------------
require(OpenMx)
mxOption(key='Number of Threads', value=parallel::detectCores()) #now
Sys.setenv(OMP_NUM_THREADS=parallel::detectCores()) #before
library(OpenMx)
library(dplyr)
library(eatTools)
library(tidyverse)
library(effectsize)


#OpenMX
#-------------------------------------------------------

#Add All matrices

# Model 1 - emotions with emotiuon metrics and purturbations
omxDetectCores() # cores available
getOption('mxOptions')$"Number of Threads" # cores used by OpenMx

data_all <- read.csv("/project/Driver_in_the_loop/data_state_space/test2.csv")
data_all <- data_all[1:10534,]
# data_all$PPG1._hav <- data_all$PPG1._hav^2
# data_all$PPG2._hav <- data_all$PPG2._hav^2
# data_all$PPG3._hav <- data_all$PPG3._hav^2
# data_all$PPG4._hav <- data_all$PPG4._hav^2
#data_all$follow_dist._hav <- data_all$follow_dist._hav^2
data_all <- select(data_all,-X)#,-Unnamed..0)
data_all <- data_all %>% 
  rename(
    #gaze_angle_x_hav = gaze_angle_x._hav,
    #gaze_angle_y_hav = gaze_angle_y._hav,
    PPG1_hav = PPG1._hav,
    PPG2_hav = PPG2._hav,
    PPG3_hav = PPG3._hav,
    PPG4_hav = PPG4._hav,
    follow_dist_hav = follow_dist._hav
  )
data_all <- scale(data_all)
data_all <- normalize(data_all)
varnames <- colnames(data_all)

amat <- mxMatrix(type = "Full", nrow = 1,
                 ncol = 1, free = TRUE,
                 values = 1, name = "A")
bmat <- mxMatrix(type = "Full", nrow = 1,
                 ncol = 1, free = TRUE,
                 values = 0.8, name = "B")

cmat <- mxMatrix(type = "Full", nrow = 2,dimnames=list(varnames[1:2], c("F1")),
                 ncol =1, free = TRUE,
                 values = .2, name = "C")
cmat$values[1,1] = 1
cmat$free[1,1] = FALSE
cmat$labels[1,1] = "C1"
cmat$values[2,2] = 1
cmat$free[2,2] = FALSE
cmat$labels[2,2] = "C2"


dmat <- mxMatrix(type = "Zero", nrow = 2,
                 ncol = 1, name = "D")
pmat <- mxMatrix(type="Diag",1,1,FALSE,1,name="P0") #cov between x and xtrue
qmat <- mxMatrix(type="Full",1,1,FALSE,1,name="Q") #cov between different factors
qmat$values[1,2] <- 0
qmat$free[1,2] <- FALSE
qmat$values[2,1] <- 0
qmat$free[2,1] <- FALSE

rmat <- mxMatrix("Diag", 2, 2, TRUE,
                 1, name = "R")

umat <-  mxMatrix(type = "Zero", nrow = 1,
                  ncol = 1, name = "u")#mxMatrix("Full", 1, 1, name = "u",label=c("carsmooth"))#,"follow_distsmooth","traffic_lightsmooth"))
xmat <- mxMatrix("Full", 1, 1, name = "x0")


#Model building and running
ssModel <- mxModel(model = "em1",
                   amat, bmat, cmat, dmat, qmat, rmat, xmat,
                   pmat, umat,
                   mxData(observed = data_all,
                          type = "raw"),
                   mxExpectationStateSpace("A", "B", "C",
                                           "D", "Q", "R", "x0", "P0", u="u"),
                   mxFitFunctionML()
)

ssRun <- mxRun(ssModel)
summary(ssRun)
ks <- mxKalmanScores(ssModel)
str(ks)
mxCheckIdentification(ssModel)





#MARSS
#-------------------------------------------------------

library(MARSS)

fulldat <- read.csv('D:/Google Drive/UVA_PhD/Projects/Structural Equation Modeling/state_space_model_1_restructured.csv')
dat <- t(cbind(fulldat$HRV_LF,fulldat$HRV_RMSSD,fulldat$PPG3_hrv,fulldat$magGyro_driverinv,fulldat$magAcce_driverinv,fulldat$entropy))
the.mean <- apply(dat, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
dat <- (dat - the.mean) * (1 / the.sigma)

covariates_per <- t(cbind(fulldat$all_vehicles_moving))
the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
# z.score the covariates covariates <- zscore(covariates)
Q <- matrix(c(1,0,0,1), 2, 2) #covariance of latent variables2*2
B <- matrix(list("b1","b2","b3","b4"), 2, 2) #transition matrix2*2
Z <- matrix(list("z11","z21","z31",0,0,0,0,0,0,"z42","z52","z62"), 6, 2) #observation matrix5*2
R <- matrix(list(0), 6, 6) #measurement covariance n*n
diag(R) <- c(1, 1,1,1,1,1) #should we allow this as a degree of freedom?
A <- matrix(0, 6, 1) #
C <- matrix(list("C11","C22"),2,1)#2*1
x <- dat # to show the relation between dat & the equations
model.list <- list(
  A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
)
kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=2000))


fulldat <- read.csv('D:/Google Drive/UVA_PhD/Projects/Structural Equation Modeling/state_space_model_1_restructured.csv')
dat <- t(cbind(fulldat$HRV_LF,fulldat$HRV_RMSSD,fulldat$PPG3_hrv,fulldat$entropy))
the.mean <- apply(dat, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
dat <- (dat - the.mean) * (1 / the.sigma)

covariates_per <- t(cbind(fulldat$all_vehicles_moving,fulldat$magGyro_driverinv,fulldat$traffic_light_moving))
the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
# z.score the covariates covariates <- zscore(covariates)
Q <- matrix(c(1,"q1","q1",1), 2, 2) #covariance of latent variables2*2
B <- matrix(list("b1","b2","b3","b4"), 2, 2) #transition matrix2*2
Z <- matrix(list("z11","z21","z31","z41","z12","z22","z32","z42"), 4, 2) #observation matrix5*2
R <- matrix(list(0), 4, 4) #measurement covariance n*n
diag(R) <- c(1, 1,1,1) #should we allow this as a degree of freedom?
A <- matrix(0, 4, 1) #
C <- matrix(list("C11","C21","C12","C22","C13","C23"),2,3)#2*1
x <- dat # to show the relation between dat & the equations
model.list <- list(
  A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
)
kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=2000))



#model 1

library(MARSS)
fulldat <- read.csv('D:/Google Drive/UVA_PhD/Projects/Structural Equation Modeling/state_space_model_1_restructured.csv')
dat <- t(cbind(fulldat$HR_bcp_mean,fulldat$AU01_c,fulldat$AU02_c,fulldat$AU06_c,fulldat$AU07_c,fulldat$AU12_c,fulldat$AU15_c,fulldat$AU25_c,fulldat$entropy))
the.mean <- apply(dat, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
dat <- (dat - the.mean) * (1 / the.sigma)

covariates_per <- t(cbind(fulldat$all_vehicles_moving,fulldat$magGyro_driverinv))
covariates_per <- t(cbind(fulldat$all_vehicles,fulldat$magGyro_driver))
the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
# z.score the covariates covariates <- zscore(covariates)
Q <- matrix(c(1,0,0,1), 2, 2) #covariance of latent variables2*2
B <- matrix(list("b1","b2","b3","b4"), 2, 2) #transition matrix2*2
Z <- matrix(list("z11","z21",0,"Z41",0,"Z61","Z71",0,0,0,"z22","z32",0,"Z52",0,0,"Z82","Z92"), 9, 2) #observation matrix5*2
R <- matrix(list(0), 9, 9) #measurement covariance n*n
diag(R) <- c(1,1,1,1,1,1,1,1,1) #should we allow this as a degree of freedom?
A <- matrix(0, 9, 1) #
C <- matrix(list("C11","C21","C12","C22"),2,2)#2*1
x <- dat # to show the relation between dat & the equations
model.list <- list(
  A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
)
kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=2000))
#model 2

library(MARSS)
fulldat <- read.csv('D:/Google Drive/UVA_PhD/Projects/Structural Equation Modeling/state_space_model_1_restructured.csv')
dat <- t(cbind(fulldat$HR_bcp_mean,fulldat$AU01_c,fulldat$AU02_c,fulldat$AU06_c,fulldat$AU07_c,fulldat$AU12_c,fulldat$AU15_c,fulldat$AU25_c,fulldat$entropy))
the.mean <- apply(dat, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
dat <- (dat - the.mean) * (1 / the.sigma)

covariates_per <- t(cbind(fulldat$all_vehicles_moving,fulldat$magGyro_driverinv))
the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
# z.score the covariates covariates <- zscore(covariates)
Q <- matrix(c(1,"q2","q2",1), 2, 2) #covariance of latent variables2*2
B <- matrix(list("b1","b2","b3","b4"), 2, 2) #transition matrix2*2
Z <- matrix(list("z11","z21",0,"Z41",0,"Z61","Z71",0,0,0,"z22","z32",0,"Z52",0,0,"Z82","Z92"), 9, 2) #observation matrix5*2
R <- matrix(list(0), 9, 9) #measurement covariance n*n
diag(R) <- c(1,1,1,1,1,1,1,1,1) #should we allow this as a degree of freedom?
A <- matrix(0, 9, 1) #
C <- matrix(list("C11","C21","C12","C22"),2,2)#2*1
x <- dat # to show the relation between dat & the equations
model.list <- list(
  A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
)
kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=2000))
MARSSparamCIs(kemfit,hessian.fun='optim',silent=FALSE)


#model 3
library(MARSS)
fulldat <- read.csv('D:/Google Drive/UVA_PhD/Projects/Structural Equation Modeling/state_space_model_1_restructured.csv')
dat <- t(cbind(fulldat$HR_bcp_mean,fulldat$AU01_c,fulldat$AU02_c,fulldat$AU06_c,fulldat$AU07_c,fulldat$AU12_c,fulldat$AU15_c,fulldat$AU25_c,fulldat$entropy))
the.mean <- apply(dat, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
dat <- (dat - the.mean) * (1 / the.sigma)

covariates_per <- t(cbind(fulldat$all_vehicles_moving,fulldat$magGyro_driverinv))
the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
# z.score the covariates covariates <- zscore(covariates)
Q <- matrix(c(1), 1, 1) #covariance of latent variables2*2
B <- matrix(list("b1"), 1, 1) #transition matrix2*2
Z <- matrix(list("z11","z21","Z31","Z41","Z51","Z61","Z71","Z81","Z91"), 9, 1) #observation matrix5*2
R <- matrix(list(0), 9, 9) #measurement covariance n*n
diag(R) <- c(1,1,1,1,1,1,1,1,1) #should we allow this as a degree of freedom?
A <- matrix(0, 9, 1) #
C <- matrix(list("C11","C12"),1,2)#2*1
x <- dat # to show the relation between dat & the equations
model.list <- list(
  A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
)
kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=2000))
#kem.with.CIs.from.hessian <- MARSSparamCIs(kemfit)



#Run model 2 for each section 

library(MARSS)
library(SparkR)

data_out <- setNames(data.frame(matrix(ncol = 25, nrow = 1)), c( "Z.z11", "Z.z21" ,"Z.Z41" ,"Z.Z61" ,"Z.Z71", 
                                                                 "Z.z22" ,"Z.z32" ,"Z.Z52" ,"Z.Z82" ,"Z.Z92",
                                                                 "B.b1"  ,"B.b2"  ,"B.b3"  ,"B.b4"  ,"U.X1" ,
                                                                 "U.X2"  ,"Q.1"  , "Q.q2"  ,"x0.X1" ,"x0.X2",
                                                                "C.C11" ,"C.C21", "C.C12" ,"C.C22","name"))

for (i in list.files(path='D:/Google Drive/UVA_PhD/Projects/Structural Equation Modeling/sectionwise_files/third method/')){
  if (endsWith(i,"csv")){
    print(i)
    fulldat <- read.csv(paste('D:/Google Drive/UVA_PhD/Projects/Structural Equation Modeling/sectionwise_files/third method/',i,sep=""))
    dat <- t(cbind(fulldat$HR_bcp_mean,fulldat$AU01_c,fulldat$AU02_c,fulldat$AU06_c,fulldat$AU07_c,fulldat$AU12_c,fulldat$AU15_c,fulldat$AU25_c,fulldat$entropy))
    the.mean <- apply(dat, 1, mean, na.rm = TRUE)
    the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
    dat <- (dat - the.mean) * (1 / the.sigma)
  
    covariates_per <- t(cbind(fulldat$all_vehicles_moving,fulldat$magGyro_driver))
    the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
    the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
    covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
    # z.score the covariates covariates <- zscore(covariates)
    Q <- matrix(c(1,"q2","q2",1), 2, 2) #covariance of latent variables2*2
    B <- matrix(list("b1","b2","b3","b4"), 2, 2) #transition matrix2*2
    Z <- matrix(list("z11","z21",0,"Z41",0,"Z61","Z71",0,0,0,"z22","z32",0,"Z52",0,0,"Z82","Z92"), 9, 2) #observation matrix5*2
    R <- matrix(list(0), 9, 9) #measurement covariance n*n
    diag(R) <- c(1,1,1,1,1,1,1,1,1) #should we allow this as a degree of freedom?
    A <- matrix(0, 9, 1) #
    C <- matrix(list("C11","C21","C12","C22"),2,2)#2*1
    x <- dat # to show the relation between dat & the equations
    model.list <- list(
      A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
    )
    kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=6000))
    data_out <- rbind(data_out,c(unname(kemfit$coef),i))
    data_out
    }

}




#Run model 2 for rolling 

library(MARSS)
library(SparkR)

results <- read.csv("C:/Users/Arsalan/Documents/all_results_new_07112021.csv")


for (j in c(2,3,9,10,12,14,16,17,18,19,20,22)){
  data_out <- setNames(data.frame(matrix(ncol = 28, nrow = 1)), c( "Z.z11", "Z.z21" ,"Z.Z41" ,"Z.Z61" ,"Z.Z71", 
                                                                   "Z.z22" ,"Z.z32" ,"Z.Z52" ,"Z.Z82" ,"Z.Z92",
                                                                   "B.b1"  ,"B.b2"  ,"B.b3"  ,"B.b4"  ,"U.X1" ,
                                                                   "U.X2"  ,"Q.1"  , "Q.q2"  ,"x0.X1" ,"x0.X2",
                                                                   "C.C11" ,"C.C21", "C.C12" ,"C.C22","name","errors"
                                                                   ,"loglike","model_name"))
  data_out_loglike = setNames(data.frame(matrix(ncol = 2, nrow = 3)), c( "model_name","loglike"))
  path = paste("H:/",toString(j),"/no_overlap/",sep="")
  for (i in list.files(path=path)){
    #if (!( i %in% results$name)){
      if (endsWith(i,".csv")){
        print(i)
        fulldat <- read.csv(paste(path,i,sep=""))
        fulldat$magGyro = (fulldat$X_Gyro**2+fulldat$Y_Gyro**2+fulldat$Z_Gyro**2)**0.5
        dat <- t(cbind(fulldat$HR_bcp_mean,fulldat$AU01_c,fulldat$AU02_c,fulldat$AU06_c,fulldat$AU07_c,fulldat$AU12_c,fulldat$AU15_c,fulldat$AU25_c,fulldat$entropy))
        the.mean <- apply(dat, 1, mean, na.rm = TRUE)
        the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
        dat <- (dat - the.mean) * (1 / the.sigma)
        
        covariates_per <- t(cbind(fulldat$all_vehicles,fulldat$magGyro))
        the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
        the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
        covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
        # z.score the covariates covariates <- zscore(covariates)
        Q <- matrix(c(1,"q2","q2",1), 2, 2) #covariance of latent variables2*2
        B <- matrix(list("b1","b2","b3","b4"), 2, 2) #transition matrix2*2
        Z <- matrix(list("z11","z21",0,"Z41",0,"Z61","Z71",0,0,0,"z22","z32",0,"Z52",0,0,"Z82","Z92"), 9, 2) #observation matrix5*2
        R <- matrix(list(0), 9, 9) #measurement covariance n*n
        diag(R) <- c(1,1,1,1,1,1,1,1,1) #should we allow this as a degree of freedom?
        A <- matrix(0, 9, 1) #
        C <- matrix(list("C11","C21","C12","C22"),2,2)#2*1
        x <- dat # to show the relation between dat & the equations
        model.list <- list(
          A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
        )
        kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=2000))
        #data_out_loglike = rbind(data_out_loglike,c("two_var_int",kemfit$logLik))
        data_out <- rbind(data_out,c(unname(kemfit$coef),i,kemfit$convergence,kemfit$logLik,"two_var_int"))
        #data_out
        
        
        
        # print(i)
        # fulldat <- read.csv(paste(path,i,sep=""))
        # fulldat$magGyro = (fulldat$X_Gyro**2+fulldat$Y_Gyro**2+fulldat$Z_Gyro**2)**0.5
        # dat <- t(cbind(fulldat$HR_bcp_mean,fulldat$AU01_c,fulldat$AU02_c,fulldat$AU06_c,fulldat$AU07_c,fulldat$AU12_c,fulldat$AU15_c,fulldat$AU25_c,fulldat$entropy))
        # the.mean <- apply(dat, 1, mean, na.rm = TRUE)
        # the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
        # dat <- (dat - the.mean) * (1 / the.sigma)
        # 
        # covariates_per <- t(cbind(fulldat$all_vehicles,fulldat$magGyro))
        # the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
        # the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
        # covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
        # # z.score the covariates covariates <- zscore(covariates)
        # Q <- matrix(c(1,0,0,1), 2, 2) #covariance of latent variables2*2
        # B <- matrix(list("b1","b2","b3","b4"), 2, 2) #transition matrix2*2
        # Z <- matrix(list("z11","z21",0,"Z41",0,"Z61","Z71",0,0,0,"z22","z32",0,"Z52",0,0,"Z82","Z92"), 9, 2) #observation matrix5*2
        # R <- matrix(list(0), 9, 9) #measurement covariance n*n
        # diag(R) <- c(1,1,1,1,1,1,1,1,1) #should we allow this as a degree of freedom?
        # A <- matrix(0, 9, 1) #
        # C <- matrix(list("C11","C21","C12","C22"),2,2)#2*1
        # x <- dat # to show the relation between dat & the equations
        # model.list <- list(
        #   A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
        # )
        # kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=6000))
        # data_out_loglike = rbind(data_out_loglike,c("two_var_non_int",kemfit$logLik))
        # #data_out <- rbind(data_out,c(unname(kemfit$coef),i,kemfit$convergence,kemfit$logLik,"two_var_non_int"))
        # #data_out
        # 
        # 
        # print(i)
        # fulldat <- read.csv(paste(path,i,sep=""))
        # fulldat$magGyro = (fulldat$X_Gyro**2+fulldat$Y_Gyro**2+fulldat$Z_Gyro**2)**0.5
        # dat <- t(cbind(fulldat$HR_bcp_mean,fulldat$AU01_c,fulldat$AU02_c,fulldat$AU06_c,fulldat$AU07_c,fulldat$AU12_c,fulldat$AU15_c,fulldat$AU25_c,fulldat$entropy))
        # the.mean <- apply(dat, 1, mean, na.rm = TRUE)
        # the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
        # dat <- (dat - the.mean) * (1 / the.sigma)
        # 
        # covariates_per <- t(cbind(fulldat$all_vehicles,fulldat$magGyro))
        # the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
        # the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
        # covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
        # # z.score the covariates covariates <- zscore(covariates)
        # Q <- matrix(c(1), 1, 1) #covariance of latent variables2*2
        # B <- matrix(list("b1"), 1, 1) #transition matrix2*2
        # Z <- matrix(list("z11","z21","Z31","Z41","Z51","Z61","Z71","Z81","Z91"), 9, 1) #observation matrix5*2
        # R <- matrix(list(0), 9, 9) #measurement covariance n*n
        # diag(R) <- c(1,1,1,1,1,1,1,1,1) #should we allow this as a degree of freedom?
        # A <- matrix(0, 9, 1) #
        # C <- matrix(list("C11","C12"),1,2)#2*1
        # x <- dat # to show the relation between dat & the equations
        # model.list <- list(
        #   A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
        # )
        # kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=6000))
        # data_out_loglike = rbind(data_out_loglike,c("one_var",kemfit$logLik))
        # #data_out <- rbind(data_out,c(unname(kemfit$coef),i,kemfit$convergence,kemfit$logLik,"one_var"))
        # #data_out
      }
  }


write.csv(data_out,paste("H:/",toString(j),"/","all_results_state_space.csv",sep=""))

}




















# 
# 
# dat <- t(cbind(fulldat$HRV_LF,fulldat$HRV_RMSSD,fulldat$PPG3_hrv,fulldat$entropy))
# the.mean <- apply(dat, 1, mean, na.rm = TRUE)
# the.sigma <- sqrt(apply(dat, 1, var, na.rm = TRUE))
# dat <- (dat - the.mean) * (1 / the.sigma)
# 
# covariates_per <- t(cbind(fulldat$all_vehicles_moving,fulldat$magGyro_driverinv,fulldat$magAcce_driverinv))
# the.mean <- apply(covariates_per, 1, mean, na.rm = TRUE)
# the.sigma <- sqrt(apply(covariates_per, 1, var, na.rm = TRUE))
# covariates_per <- (covariates_per - the.mean) * (1 / the.sigma)
# # z.score the covariates covariates <- zscore(covariates)
# Q <- matrix(c("q1"), 1, 1) #covariance of latent variables2*2
# B <- matrix(list("b1"), 1, 1) #transition matrix1*1
# Z <- matrix(list("z11","z21","z31","z41"), 4, 1) #observation matrix5*2
# R <- matrix(list(0), 4, 4) #measurement covariance n*n
# diag(R) <- c(1, 1,1,1) #should we allow this as a degree of freedom?
# A <- matrix(0, 4, 1) #
# C <- matrix(list("C11","C22","C33"),1,3)#2*1
# x <- dat # to show the relation between dat & the equations
# model.list <- list(
#   A=A, B = B,C=C, Q = Q, Z = Z, R = R,c=covariates_per,tinitx=1
# )
# kemfit <- MARSS(x, model = model.list,inits = list(x0=0.3),control =list(maxit=2000))
# 
# # parvec <- coef(kemfit, type = "vector")
# df <- tsSmooth(kemfit)# For getting the smoothened estimates
# plot(df$.estimate)
# print(kemfit, what = "Z")
# print(kemfit, what = "B")
# print(kemfit, what = "Q")
# print(kemfit, what= "R")
# print(kemfit, what= "C")
# result<-cbind(df$.estimate, df$.se, fulldat$heartrate, fulldat$coinprob)
# write.csv(result, "result.csv")
# 
# 
# 
# Z_vals <- list("z11", 0,"Z21","Z22","Z31","Z32","Z41","Z42","Z51","Z52","Z61","Z62","Z71","Z72","Z81","Z82")
# ZZ <- matrix(Z_vals, nrow = 8, ncol = 2, byrow = TRUE)
# 
# ## 'aa' is the offset/scaling
# aa <- "zero"
# ## 'DD' and 'd' are for covariates
# DD <- "zero"  # matrix(0,mm,1)
# dd <- "zero"  # matrix(0,1,wk_last)
# ## 'RR' is var-cov matrix for obs errors
# RR <- "diagonal and unequal"
# 
# ## number of processes
# mm <- 2
# ## 'BB' is identity: 1's along the diagonal & 0's elsewhere
# BB <- "identity"  # diag(mm)
# ## 'uu' is a column vector of 0's
# uu <- "zero"  # matrix(0, mm, 1)
# ## 'CC' and 'cc' are for covariates
# CC <- "zero"  # matrix(0, mm, 1)
# cc <- "zero"  # matrix(0, 1, wk_last)
# ## 'QQ' is identity
# QQ <- "identity"  # diag(mm)
# 
# ## list with specifications for model vectors/matrices
# mod_list <- list(Z = ZZ, A = aa, D = DD, d = dd, R = RR, B = BB, 
#                  U = uu, C = CC, c = cc, Q = QQ)
# ## list with model inits
# init_list <- list(x0 = matrix(rep(0, mm), mm, 1))
# ## list with model control parameters
# con_list <- list(maxit = 3000, allow.degen = TRUE)
# 
# dfa_1 <- MARSS(y = dat, model = mod_list, inits = init_list, 
#                control = con_list)
# 
