#!/bin/bash

# à modifier en indiquant où est installé simgrid
#SIMGRID=/usr
#export PATH=${SIMGRID}/bin:${PATH}

# il faut copier le répertoire "archis" à votre racine
alias Smpirun="smpirun -hostfile ${PWD}/archis/cluster_hostfile.txt -platform ${PWD}/archis/cluster_crossbar.xml"
