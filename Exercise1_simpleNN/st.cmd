require calc

# Default input range
# XMIN=0, XMAX=100

# Hidden neuron weights
# W1=-20,B1=6.6
# W2=20,B2=-6.6
# W3=20,B3=-13.2

# Output layer weights
# L1=6,L2=0,L3=0,LB=-3
# M1=0,M2=6,M3=-6,MB=-3
# H1=0,H2=0,H3=6, HB=-3

# Load database with weights above
dbLoadRecords("nn.db", "P=SYS:,W1=-20,B1=6.6,W2=20,B2=-6.6,W3=20,B3=-13.2,L1=6,L2=0,L3=0,LB=-3,M1=0,M2=6,M3=-6,MB=-3,H1=0,H2=0,H3=6, HB=-3")

#Start IOC
iocInit()
