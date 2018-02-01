from __future__ import division                              #What does it mean exactly? 
from ROOT import gROOT, TH2D,TH1D, TF2, TCanvas, TPad, TRandom,  TFile, TColor, TStyle, TSpectrum
gROOT.Reset()
import sys
import os
import os.path
import glob
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
def input_matrix(k, H):
    size_H = H.shape
    rim_0 = int(floor(size_H[0]/2.))
    F = np.zeros((k+rim_0,k+rim_0))
    return F 
    
# Gaussian filter function --- F and H are numpy matrices :     
def filter(F, H): 
    size_F = F.shape
    size_H = H.shape 
    l = size_H[0]
    k = int(floor(l/2))
    rim_0 = k 
    dim = size_F[0]-2*rim_0                                        # I take the rim off. 
    G = np.zeros((dim,dim))
    F_tilde = np.zeros((l, l))
    for i in range(0,dim):                                    # Helgas Rat: durch Matrix Mulitplikation ersetzen. 
        for j in range(0,dim): 
            for u in range(0,l): 
                for v in range(0,l): 
                    F_tilde[u,v] = F[i+k-u + rim_0,j+k-v + rim_0] 
            G[i,j] = np.sum(H*F_tilde)
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
def read_histo(histo, H):                                   # The kernel H is also asked for, so I can add and adequate rim of zero entries to the matrix for the Gaussian filter of the rim elements. 
    size_H = H.shape
    rim_0 = int(floor(size_H[0]/2.))
    bin_x_min = histo.GetXaxis().GetFirst()
    bin_x_max = histo.GetXaxis().GetLast()
    bin_y_min = histo.GetYaxis().GetFirst()
    bin_y_max = histo.GetYaxis().GetLast()
    M = np.zeros((bin_x_max + 2*rim_0,bin_y_max + 2*rim_0))         # 2*rim_0 is a rim above and below, on the right and on the left. 
    #print(M)
    for i in range(rim_0, bin_x_max + rim_0): 
        for j in range(rim_0, bin_y_max + rim_0): 
            M[i,j] = histo.GetBinContent(i-rim_0+1, j-rim_0+1)
    #print(M) 
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

def fine_binning(root_dir, save_files_dir): 
    # Set up canvas : 
    w = 1400 
    h =  700
    can  = TCanvas("can", "can you fill boxes with boxes?  ", w, h)
    pad1 = TPad( 'pad1', 'Original histogram', 0.05, 0.05, 0.50, 0.95, -2)
    pad2 = TPad( 'pad2', 'Finer-binned histogram', 0.50, 0.05, 0.95, 0.95, -2 )

    #TPad( name, title, xlow, ylow, xup, yup, color = -1, bordersize = -1, bordermode = -2 ) 	

    # Draw the pads : 
    pad1.Draw()
    pad2.Draw()
 

    # Get the data from root file : 
    file = TFile(root_dir, "READ")
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
    histo_fine = TH2D("histo_fine", "Fine-binned histogram of annulus ", bin_x_max*grid, x_min, x_max, bin_y_max*grid, y_min, y_max) 

    print(" time duration before get_histo_content : ", time.time() - start_time ,"s")

    # Get histogram content :                                                                   # See function above for details on how to get the histo content.
    (x_histo, y_histo, z_histo) = get_histo_content(histo)    

    print(" time duration before fine_rebin : ", time.time() - start_time ,"s")
    # Fill the higher resolution histogram : 
    histo_fine = fine_rebin(histo_fine, x_histo, y_histo, z_histo, grid) 

    # Draw the histograms  
    pad1.cd()                               # Here the 'directory' is changed to pad1 defined above. 
    histo.Draw('colz')
    pad2.cd()
    histo_fine.Draw('colz')
    can.Update()                            # Shows the histogram in the Canvas without having to click on it. 

    print(" time duration before writing files into root file : ", time.time() - start_time ,"s")

    # Open a ROOT file and save the three histograms. 
    my_file = TFile( save_files_dir, 'RECREATE' )
    histo.Write()
    histo_fine.Write()
    my_file.Close()

    print(" time duration : ", time.time() - start_time ,"s")

root_dir       = '../root_files/raw_data/*_annulus.root'
save_files_dir = '../root_files/fine_binned/'


#for files in os.listdir(root_dir): 
#    if pattern in files: 
#        fine_binning(root_dir, save_files_dir)


for fname in glob.glob(root_dir):
    print(fname)
    list = fname.split('/')
    raw_name = list[3].split('_')
    save_files_dir_filename = '../root_files/fine_binned/' + raw_name[0] + '_fine_binned_annulus.root'
    number = float(raw_name[0])
    if number > 98500: 
        fine_binning(fname, save_files_dir_filename)
        print(number)
        print(save_files_dir)
