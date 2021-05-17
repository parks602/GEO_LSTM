#!/bin/sh
# PATH
MAIN="/home/onlyred/KMA/TEST_GEO"
SHEL="${MAIN}/SHEL"
SRCS="${MAIN}/SRCS"
DABA="${MAIN}/DABA"
DAIN="${MAIN}/DAIN"
DAOU="${MAIN}/DAOU"
# OPTION
SDATE="2019020100"
EDATE="2019022823"
MODE="train"
VARI="T3H"
WS=8         # window size
NK=10        # nearest K
BS=16        # Batchsize
EP=500       # epochs
LR=0.001     # learning rate

# DATA
DATA=${DAIN}/${VARI}_aws+buoy_spatio_temporal.csv
INFO=${DAIN}/${VARI}_aws+buoy_stninfo.csv
ODIR=${DAOU}/${VARI}_OBS_TEST_2
MESH=${DAIN}/mesh_1km.csv
MODL=${ODIR}/checkpoint.pt

python ${SRCS}/main.py --dataf ${DATA} --infof ${INFO} --targf ${INFO} \
	               --windowSize ${WS} --nearestK ${NK} \
		       --epochs ${EP} --lr ${LR} --batchSize ${BS} \
		       --opath ${ODIR} --mode ${MODE} \
		       --sdate ${SDATE} --edate ${EDATE} --var ${VARI}
