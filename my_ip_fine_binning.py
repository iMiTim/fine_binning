from __future__ import division                              #What does it mean exactly? 
from ROOT import gROOT, TH2D,TH1D, TF2, TCanvas, TPad, TRandom,  TFile, TColor, TStyle, TSpectrum
gROOT.Reset()
import sys
import numpy as np
import time 
from math import sin 
from math import floor 
from math import exp
from math import sqrt
import matplotlib.pyplot as plt 

start_time = time.time()

# Function section : 

# 2D Gaussian function : 
def gauss(sigma, x, y): 
    pi = 3.14159265359
    return (1./(2.*pi*sigma*sigma))*exp(-(x*x+y*y)/(2.*sigma*sigma))


# function for Kernel generation : 
# If a Kernel with a bigger width is needed, I have to sample more points and add them up to get a better approximation of the integral. 
# Alternatively, I just use the same simple Kernel successively : sigma_[i] = sigma_[i-1]*sqrt(2), with sigma_[0]=sigma  and i = 0, 1, 2 ... n
def kernel_generator(sigma, px_width, accuracy):
    print("sigma, px_width : ", sigma, px_width)
    dx = floor(sigma/px_width)
    print("floor(sigma/px_width : ", dx)
    dy = dx                                                                         # For now, a symmetric Gaussian Kernel is sufficient. 
    if dx < 1 : 
        print("Error : Please choose a larger sigma! ")
        return 1
    dx = dx * 2*accuracy + 1
    dy = dx        
    m = int(dx)
    n = int(dy)                                                                 # For now, a symmetric Gaussian Kernel is sufficient. 
    s = int(floor(dx/2))
    H = np.zeros((m,n))
    print("m, n, s : ", m, n, s)
    for i in range(int(m)):
        for j in range(int(n)):
            H[i,j] = (gauss(sigma , (i-s)*px_width, (j-s)*px_width))
    return H

# Input matrix declaration : 
def input_matrix(k):
    F = np.zeros((k+2,k+2))
    F[(k+2)/2,(k+2)/2] = 1.
    F[k-2,k-2] = 1. 
    return F 
    
# Gaussian filter function :     
def filter(F, H): 
    size = F.shape
    k = size[0]-2
    G = np.zeros((k,k))
    for i in range(0,k):                                    # Helgas Rat: durch Matrix Mulitplikation ersetzen. 
        for j in range(0,k): 
            for u in range(0,3): 
                for v in range(0,3):
                    G[i,j] += H[u,v]*F[i-(u-2),j-(v-2)]
    #print(H)
    #print(F)
    #print(G)
    return G

# ****************

# Get the histogram content : 
def get_histo_content(histo): 
    x = []
    y = []
    z = [] 
    bin_x_min = histo.GetXaxis().GetFirst()
    bin_x_max = histo.GetXaxis().GetLast()
    bin_y_min = histo.GetYaxis().GetFirst()
    bin_y_max = histo.GetYaxis().GetLast()
    for i in range(bin_x_min, bin_x_max + 1): 
        tempX = histo.GetXaxis().GetBinCenter(i)
        for j in range(bin_y_min, bin_y_max + 1): 
            tempY = histo.GetYaxis().GetBinCenter(j)
            tempZ = histo.GetBinContent(i,j)
            if (tempZ != 0.) : 
                x.append(tempX) 
                y.append(tempY)
                z.append(tempZ)
    return (x, y, z)

# Fill the more fine-grained histogram. 
def fine_rebin(histo, x, y, z, grid): 
    Lz = len(z)
    grain = histo.GetXaxis().GetBinWidth(1)
    if (grid % 2)==1 : 
        print("Uneven number ")
        dp = floor(grid/2.)
        for i in range(Lz): 
            tempX = x[i] - dp*grain
            tempY = y[i] - dp*grain
            tempZ = z[i]/(grid*grid)
            for j in range(grid): 
                for k in range(grid): 
                    histo.Fill(tempX + grain*k, tempY + grain*j, tempZ) #(j+1)*(k+1)*
    return histo

# Get histo integral : 
def get_integral(histo): 
    bin_x_min = histo.GetXaxis().GetFirst()
    bin_x_max = histo.GetXaxis().GetLast()
    bin_y_min = histo.GetYaxis().GetFirst()
    bin_y_max = histo.GetYaxis().GetLast()

    int_histo = histo.Integral(bin_x_min, bin_x_max, bin_y_min, bin_y_max)
    print( bin_x_min, bin_x_max, bin_y_min, bin_y_max)
    return int_histo

# Read histogram into matrix : 
def read_histo(histo): 
    bin_x_min = histo.GetXaxis().GetFirst()
    bin_x_max = histo.GetXaxis().GetLast()
    bin_y_min = histo.GetYaxis().GetFirst()
    bin_y_max = histo.GetYaxis().GetLast()
    M = np.zeros((bin_x_max + 2,bin_y_max + 2))
    print(M)
    for i in range(bin_x_max): 
        for j in range(bin_y_max): 
            M[i,j] = histo.GetBinContent(i, j)
    print(M) 
    return M 

# Feed matrix back into histogram : 
def write_histo(M, histo): 
    size  = M.shape
    dim_x = size[0]
    dim_y = size[1]
    for i in range(dim_x): 
        for j in range(dim_y): 
            w = M[i,j]
            #print('i + 1, j + 1, w : ', i + 1, j + 1, w)
            x = histo.GetXaxis().GetBinCenter(i+1)
            y = histo.GetYaxis().GetBinCenter(j+1)
            histo.Fill(x, y, w)
    return histo

############################################################################################################
                                    ### Main body of the code : ###
############################################################################################################


# Set up canvas : 
w = 1400 
h =  700
can  = TCanvas("can", "can you fill boxes with boxes?  ", w, h)
pad1 = TPad( 'pad1', 'Original histogram', 0.05, 0.05, 0.45, 0.95, -2)
pad2 = TPad( 'pad2', 'Finer-binned histogram', 0.55, 0.05, 0.95, 0.95, -2 )
#TPad( name, title, xlow, ylow, xup, yup, color = -1, bordersize = -1, bordermode = -2 ) 	

# Draw the pads : 
pad1.Draw()
pad2.Draw()

# Get the data from root file : 
file = TFile("../root_files/78722_annulus.root", "READ")
histo = file.Get("ImgFull5THDetCMOS_0")
# Properties of the histogram from file : 
bin_x_min = histo.GetXaxis().GetFirst()
bin_x_max = histo.GetXaxis().GetLast()
bin_y_min = histo.GetYaxis().GetFirst()
bin_y_max = histo.GetYaxis().GetLast()
bin_x_width = histo.GetXaxis().GetBinWidth(bin_x_min)
bin_y_width = histo.GetYaxis().GetBinWidth(bin_y_min)
x_min = histo.GetXaxis().GetBinCenter(bin_x_min)-bin_x_width/2.
x_max = histo.GetXaxis().GetBinCenter(bin_x_max)+bin_x_width/2.
y_min = histo.GetYaxis().GetBinCenter(bin_y_min)-bin_y_width/2.
y_max = histo.GetYaxis().GetBinCenter(bin_y_max)+bin_y_width/2.
print("x_min, x_max : ", x_min, x_max)
print("y_min, y_max : ", y_min, y_max)
print("bin_x_min, bin_x_max : ", bin_x_min, bin_x_max)
print("bin_y_min, bin_y_max : ", bin_y_min, bin_y_max)          

# Constant and histogram definitions for histogram : 
grid = 3
histo_gauss = TH2D("histo_gauss", "Blurred histogram of annulus ", bin_x_max, x_min, x_max, bin_y_max, y_min, y_max)
histo_fine = TH2D("histo_fine", "Fine-binned histogram of annulus ", bin_x_max*grid, x_min, x_max, bin_y_max*grid, y_min, y_max) 
histo_fine_gauss = TH2D("histo_fine_gauss", "Blurred fine-binned histogram of annulus ", bin_x_max*grid, x_min, x_max, bin_y_max*grid, y_min, y_max) 


# Get histogram content :                                                                   # See function above for details on how to get the histo content.
(x_histo, y_histo, z_histo) = get_histo_content(histo)    

# Fill the higher resolution histogram : 
histo_fine = fine_rebin(histo_fine, x_histo, y_histo, z_histo, grid) 

# Read histogram content into matrix : 
Histo = read_histo(histo)
Histo_fine = read_histo(histo_fine)

print(" time duration before gaussian blur : ", time.time() - start_time ,"s")

# Generate Kernel : 
H = kernel_generator(1., 1., 5.)
#kernel_generator(sigma, px_width, accuracy)  an accuracy of 2 is we have at least two standard deviations of the gaussian covered. 
sum_H = np.sum(H)
print("sum_H : ", sum_H)


# Apply Gaussian filter : 
Histo_gauss = filter(Histo, H)
Histo_fine_gauss = filter(Histo_fine, H)

# Write the matrix content into a new histogram : 
histo_gauss = write_histo(Histo_gauss, histo_gauss)
histo_fine_gauss = write_histo(Histo_fine_gauss, histo_fine_gauss)


# Testing : Is the histo integral conserved? 
int_histo = get_integral(histo)
int_histo_fine = get_integral(histo_fine)
print("int_histo : ", int_histo)
print("int_histo_fine : ", int_histo_fine)


# Draw the histograms  
pad1.cd()                               # Here the 'directory' is changed to pad1 defined above. 
histo.Draw('colz')
pad2.cd()
histo_fine.Draw('colz')
can.Update()                            # Shows the histogram in the Canvas without having to click on it. 


print(" time duration : ", time.time() - start_time ,"s")