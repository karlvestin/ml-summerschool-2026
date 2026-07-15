require calc

# Default input range
# XMIN=0, XMAX=100

# Hidden layer weights
# W1=-50,R11=1,R12=50,R13=-1,B1=-2.5
# W2=1,  R21=0.1,R22=0.1,R23=-0.1,B2=0.001
# W3=50, R31=-1,R32=-50,R33=1,B3=-2.5

# Output layer weights
# FALLING: L1=6,L2=-3,L3=-6,LB=0
# STABLE:  M1=-4,M2=6,M3=-4,MB=0
# RISING:  H1O=-6,H2O=-3,H3O=6,HB=0

# Load database with weights above
dbLoadRecords("rnn.db", "P=SYS:,W1=-50,R11=1,R12=50,R13=-1,B1=-2.5,W2=1,R21=0.1,R22=0.1,R23=-0.1,B2=0.001,W3=50, R31=-1,R32=-50,R33=1,B3=-2.5")

#Start IOC
iocInit()
