require calc

# Default input range
# XMIN=0, XMAX=100

# Default hidden neuron weights
# W1=20,B1=-6.6
# W2=20,B2=-13.2
# W3=0,B3=0

# Output layer weights
# L1=-6,L2=0,L3=0,LB=3
# M1=6,M2=-6,M3=0,MB=-3
# H1=0,H2=6,H3=0, HB=-3

# Load database with weights above
dbLoadRecords("nn.db", "P=SYS:")

#Start IOC
iocInit()
