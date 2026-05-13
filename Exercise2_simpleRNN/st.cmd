require calc

# Default input range
# XMIN=0, XMAX=100

# Default trend parameters
# EPS=0.02      deadband for STABLE trend
# K=100         trend sensitivity
# R1=1.5        recurrent memory for FALLING neuron
# R2=1.5        recurrent memory for STABLE neuron
# R3=1.5        recurrent memory for RISING neuron

# Hidden layer equations
# H1 = sigmoid(K*(-DELTA-EPS)      + R1*H1_PREV)    # FALLING
# H2 = sigmoid(K*( EPS-ABS(DELTA)) + R2*H2_PREV)    # STABLE
# H3 = sigmoid(K*( DELTA-EPS)      + R3*H3_PREV)    # RISING

# Output layer weights
# FALLING: L1=6,L2=-3,L3=-6,LB=0
# STABLE:  M1=-4,M2=6,M3=-4,MB=0
# RISING:  H1O=-6,H2O=-3,H3O=6,HB=0

# Load database with weights above
dbLoadRecords("rnn.db", "P=SYS:")

#Start IOC
iocInit()
