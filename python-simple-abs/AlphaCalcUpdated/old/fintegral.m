function  I=fintegral(beta,Delta,Ec,P)

f0=integral(int(beta,xi,Delta),xi,0,Ec)-P

end