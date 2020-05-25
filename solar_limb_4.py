# -*- coding: utf-8 -*-

'''
Yr3 Astrophysics Lab - Solar limb darkening experiment
Code for fitting pixel data of solar disk with limb-darkening function 
and extracting the coefficients
'''

import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit


################################ Inputs ######################################

# Raw data file
filename = 'solar_plot_5.txt' 

# Data selection - 
# Select the method of cropping data beyond solar disk
manual_crop = False

# When manual_crop is switched on, enter the pixel boundaries to be cropped 
x_crop_lower_limit = 50.
x_crop_upper_limit = 1750. 

# When manual_crop is switched off, execute auto-crop algorithm with grad analysis
# Set number of sections for data binning 
num_sections = 50
# Set gradient limit for removing data beyond solar disk 
std_limit = 0.9   # in units of std 
# NB. Auto-crop might not work when data has large fluctuations and short background flux

# Coefficient search - 
# Set range and step size of coeff a, b and c to be scanned 
# Range defines the boundaries of the search region around the guess values
# generated from init_coeff function
range_a = 0.06
range_b = 0.5 
range_c = 0.05
step_a = 0.01
step_c = 0.01
# NB. Bigger range and smaller step size give more precise results but longer 
#     computation time 


##############################################################################

def data_import(filename):
    
    processed_data = []
    with open(filename) as f:
        data = f.readlines()   
        for i in range(len(data)):
            entry = data[i].split()
            processed_data.append(entry)
            
    x_data = [float(p[0]) for p in processed_data]
    y_data = [float(p[1]) for p in processed_data]
    
    return x_data, y_data


def plot_graph(x_data,y_data,y_fitted):
    
    fig1 = plt.figure(figsize=(13,6)) 
    ax1 = fig1.add_subplot(111)
    ax1.plot(x_data,y_data,'.',color='royalblue',markersize=1) 
    ax1.plot(x_data,y_fitted,'-',color='red',markersize=1) 
    plt.show()
    
    return 


def darkening_function(r, a, b, c):
    '''
    Solar limb darkening function with coefficients a, b and c
    P = a + b*miu + c*miu where:
        miu = cos(theta) = (1 - r^2)^(1/2) is the Optical depth,    
        a, b, c are constants to be determined,
        and a + b + c = 1
    '''
    miu = np.sqrt(1 - r**2)
    
    return a + b* miu + c* miu**2


def quadratic_function(x, graph_a, graph_b, graph_c):
    '''
    Quadratic function with coefficients a, b and c
    Used for curve_fitting
    '''
    return graph_a + graph_b*x + graph_c*x**2


def scipy_curve_fitting(x_data,y_data):
    '''
    Fits data using scipy curve_fit for quadratic function
    USed for plotting fits on raw data and identifying max pt 
    NOTE: coefficients a, b and c generated in this function are NOT the same as 
    the limb darkening coefficients
    '''
    popt, pcov = curve_fit(quadratic_function, x_data, y_data, p0=[1,1,1])
    graph_c = popt[2]
    graph_b = popt[1]
    graph_a = popt[0]
    y_fitted = [graph_a + graph_b*x + graph_c*x**2 for x in x_data]
    
    return y_fitted


def data_binning(data_list):
    '''
    Generates the smoothed coordinates for an arbitrary data_list
    '''    
    section_length = int(len(data_list)/num_sections) 
    
    quant_data_list = []
    for n in range(0,num_sections):
        section_data = data_quantization(n,section_length,data_list)
        section_median = np.median(section_data)
        quant_data_list.append(section_median)
        
    return quant_data_list


def data_quantization(n,section_length,data_list):
    '''
    Subfunction of data_binning
    Assigns a part of raw data into a particular section n
    '''
    section_data = []
    for i,point in enumerate(data_list):  
        if i >= section_length*n and i < section_length*(n+1):            
            section_data.append(point) 

    return section_data


def disk_crop(x_data, y_data):
    '''
    Evaluates gradient of each section and identify extremes using std 
    Removing data beyond the solar disk 
    Outputs the smoothed data coords and the cropped data coords
    '''
    print('Auto-crop selected - executing subroutine disk_crop')
    
    quant_xlist = data_binning(x_data)
    quant_ylist = data_binning(y_data)
    print('len of quant_xlist', len(quant_xlist))
    
    grad_list = []
    for i in range(len(quant_xlist)-1):
        grad = (quant_ylist[i+1]-quant_ylist[i])/(quant_xlist[i+1]-quant_xlist[i])
        grad_list.append(grad)
        
    grad_mean = np.mean(grad_list)
    grad_std = np.std(grad_list) 
    
    start_index_list = []
    end_index_list = []
    for i,grad in enumerate(grad_list):
        # identify data points with extreme gradients
        if grad >= grad_mean + std_limit*grad_std:
            start_index_list.append(i+1)
        elif grad <= grad_mean - std_limit*grad_std:
            end_index_list.append(i)
    
    # The start and end quant_data index of solar disk    
    start_index = start_index_list[-1]
    end_index = end_index_list[0]
    # The start and end x values to be cropped out
    print('start_x: ', quant_xlist[start_index])
    print('end_x: ', quant_xlist[end_index])
    
    
    if quant_xlist[end_index] > quant_xlist[start_index]:
        # extract data which falls within the start and end boundaries 
        cropped_x_data = []
        cropped_y_data = []
        for coord in zip(x_data,y_data):
            if coord[0] > quant_xlist[start_index] and coord[0] <= quant_xlist[end_index]:
                cropped_x_data.append(coord[0])
                cropped_y_data.append(coord[1])
    else:
        print('Invalid crop - adjust num_sections and/or std_limit to proceed,' \
              '\n or select the manual_crop option')
        
    print('len cropped_x',len(cropped_x_data))

    return quant_xlist, quant_ylist, cropped_x_data, cropped_y_data


def manual_disk_crop(x_data, y_data):
    '''
    Removing data beyond the solar disk using the user-defined limits 
    '''
    print('Manual-crop selected - executing subroutine manual_disk_crop')
    print('start_x: ', x_crop_lower_limit)
    print('end_x: ', x_crop_upper_limit)
    cropped_x_data = []
    cropped_y_data = []
    for coord in zip(x_data,y_data):
        if coord[0] >= x_crop_lower_limit and coord[0] <= x_crop_upper_limit:
            cropped_x_data.append(coord[0])
            cropped_y_data.append(coord[1])
    
    return cropped_x_data, cropped_y_data


def data_normalization(cropped_x_data, cropped_y_data, cropped_y_fitted):
    '''
    Identifies the max point in original fitted curve
    Set this point as x=0 and y=1, rescale all data 
    then normalize the x-axis from -1 to +1
    Output rescaled data
    '''
    for coord in zip(cropped_x_data,cropped_y_fitted):
        if coord[1] == max(cropped_y_fitted):
            max_coord = coord
    print('max_coord', max_coord)

    # shift the x-axis to set midpoint as x=0  
    rescaled_x_data = [x - max_coord[0] for x in cropped_x_data]
    # normalize the y-axix to set midpoint as y=1
    rescaled_y_data = [y / max_coord[1] for y in cropped_y_data]

    # further normalize x-axis to set range as -1 to +1 
    abs_rescaled_x_data = [abs(x) for x in rescaled_x_data] # find largest x or -x       
    max_x = max(abs_rescaled_x_data)
    rescaled_x_data = [x / max_x for x in rescaled_x_data]
            
    return rescaled_x_data, rescaled_y_data


def miu_conversion(rescaled_x_data, rescaled_y_data):
    '''
    Convert x_data into miu_data for plotting 
    miu = cos(theta) = (1 - r^2)^(1/2)
    '''
    miu_data = []
    y_data_miu = []
    for coord in zip(rescaled_x_data, rescaled_y_data):
        # ensure that there is no invalid value 
        if coord[0] <= 1 and coord[0] >= -1:
            miu_data.append(np.sqrt(1 - coord[0]**2))
            y_data_miu.append(coord[1])
            
    fig2 = plt.figure(figsize=(13,6)) 
    ax2 = fig2.add_subplot(111)
    ax2.plot(miu_data, y_data_miu,'.',color='blue',markersize=1) 
    plt.show()
    
    return miu_data, y_data_miu, ax2


def init_coeff(miu_data, y_data_miu, ax2):
    '''
    Set up initial guesses for the darkening coeffs by fitting the
    miu_data against rescaled_y_data plot with linear equation
    '''
    coeff = np.polyfit(miu_data, y_data_miu,1)
    init_a = coeff[1]     # Use y-intercept to guess constant a     
    init_b = coeff[0]     # Use grad to guess constant b
    
    # Using the difference between linear fit and actual data at x=1,
    # provide a rough guess for c - 
    
    # y value of linear fit at x=1
    linear_y_x1 = coeff[0] + coeff[1] 
    # median y value of y_data_miu at around x=1 (between 0.95 and 1)
    y_data_miu_x1_group = []
    for coeff in zip(miu_data, y_data_miu):
        if coeff[0] > 0.95 and coeff[0] <= 1:
            y_data_miu_x1_group.append(coeff[1])
    y_data_miu_x1 = np.median(y_data_miu_x1_group)
    # difference between the two
    init_c = y_data_miu_x1 - linear_y_x1
    
    y_fit = [init_a + init_b*x + init_c*x**2 for x in miu_data]
    ax2.plot(miu_data, y_fit,'-',color='blue',markersize=1) 
    plt.show()
    
    print('init_a:',init_a, 'init_b:',init_b, 'init_c',init_c)  
    
    return init_a, init_b, init_c
    
    
def coeff_optimization(init_a, init_b, init_c, miu_data, y_data_miu):
    '''
    Loop through all combinations of a, b and c around init_values 
    Satisfy condition a + b + c = 1 
    '''   
    print('Begin coefficient search')
    
    coeff_comb_list = []
    chi2_list= []
    # coeff a within init_a +/- range_a
    for test_a in np.arange(init_a - range_a, init_a + range_a, step_a):
        # coeff c within init_c - range_b to 0 (negative)
        for test_c in np.arange(init_c - range_c, 0, step_c):
            # coeff b must satisfy
            test_b = 1 - test_a - test_c 
            # proceed only if b is within init_b +/- range_b
            if test_b >= init_b - range_b and test_b <= init_b + range_b :
                
                coeff_comb = [test_a, test_b, test_c]
                
                test_y_fit = [test_a + test_b*x + test_c*x**2 for x in miu_data]
                chi2 = eval_chi2(test_y_fit, y_data_miu)
                                
                coeff_comb_list.append(coeff_comb)
                chi2_list.append(chi2)
                
            else:
                print('Invalid test_b - increase range_b to proceed')
                
    return coeff_comb_list, chi2_list


def eval_chi2(y_fit, y_data):
    '''
    Subfunction of coeff_optimization
    Calculates chi2 difference between fitting curve and data 
    '''
    chi2 = 0
    for i in range(0,len(y_fit)): 
        chi2 += (y_data[i] - y_fit[i])**2 / y_fit[i]
    chi2 = chi2*(1/2) 
    
    return chi2 


def final_coeff(coeff_comb_list, chi2_list):
    '''
    Evaluates all chi2 values to obtain the most optimized coefficients
    '''
    min_chi2 = min(chi2_list)
    for coeff_comb, chi2 in zip(coeff_comb_list, chi2_list):
        if chi2 == min_chi2:
            final_coeff = coeff_comb 
            
    coeff_a = final_coeff[0]
    coeff_b = final_coeff[1]
    coeff_c = final_coeff[2]
    print('Final result - ')
    print('coeff_a:', coeff_a,'coeff_b:',coeff_b,'coeff_c:',coeff_c)
    
    return coeff_a, coeff_b, coeff_c 


def main():
    
    # Figure 1 - plot of raw data
    x_data,y_data = data_import(filename)
    y_fitted = scipy_curve_fitting(x_data, y_data)    
    plot_graph(x_data, y_data, y_fitted)
    
    # Figure 2 - plot of cropped data (only the solar disk)
    # select between auto crop or manual crop
    if manual_crop == False:
        quant_xlist, quant_ylist, cropped_x_data, cropped_y_data = disk_crop(x_data, y_data)
    elif manual_crop == True:
        cropped_x_data, cropped_y_data = manual_disk_crop(x_data, y_data)    
    else:
        print('Please choose a valid crop_data method')
    cropped_y_fitted = scipy_curve_fitting(cropped_x_data, cropped_y_data)
    plot_graph(cropped_x_data, cropped_y_data, cropped_y_fitted)
    
    # Figure 3 - plot of rescaled data
    rescaled_x_data, rescaled_y_data = data_normalization(cropped_x_data, cropped_y_data, \
                                                          cropped_y_fitted)
    rescaled_y_fitted = scipy_curve_fitting(rescaled_x_data, rescaled_y_data)
    plot_graph(rescaled_x_data, rescaled_y_data, rescaled_y_fitted)
    
    # Figure 4 - plot of miu_data against y 
    miu_data, y_data_miu, ax2 = miu_conversion(rescaled_x_data, rescaled_y_data)
    # Initial guesses for darkening coefficients, fitted 
    init_a, init_b, init_c = init_coeff(miu_data, y_data_miu, ax2)
    
    # Looping over all combinations to optimize the coefficients
    coeff_comb_list, chi2_list = coeff_optimization(init_a, init_b, init_c, \
                                                    miu_data, y_data_miu)
    coeff_a, coeff_b, coeff_c = final_coeff(coeff_comb_list, chi2_list)
    
    # Figure 5 -plot fit line with the final coeffs
    y_darkening_fit = []
    for x in rescaled_x_data:
        y_darkening_fit.append(darkening_function(x, coeff_a, coeff_b, coeff_c))
    plot_graph(rescaled_x_data, rescaled_y_data, y_darkening_fit)
        
    return 


if __name__ == '__main__':
    main()























